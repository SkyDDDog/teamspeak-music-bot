import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import {
  ChannelRecorder,
  buildRecorderFfmpegArgs,
  listRecordingFiles,
  recordingFileName,
  safeRecordingPath,
  sweepOldRecordings,
  type SpawnFn,
} from "./channel-recorder.js";
import type { Logger } from "../logger.js";

// --- @discordjs/opus stub (behavior swapped per test via `stub`) ---
const stub: {
  decode: (b: Uint8Array) => Buffer;
  encode: (b: Uint8Array) => Buffer;
} = {
  decode: () => Buffer.alloc(0),
  encode: () => Buffer.alloc(0),
};
vi.mock("@discordjs/opus", () => ({
  default: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    OpusEncoder: class {
      constructor() { return stub as any; }
    },
  },
}));

const silentLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
  child: () => silentLogger,
} as unknown as Logger;

/** Fake ffmpeg: records stdin bytes on a growing list. */
function makeFakeSpawn(written: Buffer[]): SpawnFn {
  return () => {
    const stdin = new PassThrough();
    stdin.on("data", (c: Buffer) => written.push(c));
    const fake = {
      stdin,
      kill: () => {},
      on: () => {},
      once: () => {},
      unref: () => {},
      pid: 12345,
    };
    return fake as unknown as ReturnType<SpawnFn>;
  };
}

/** PCM mono frame: constant int16 value. */
function pcmFrame(value: number): Buffer {
  const b = Buffer.alloc(1920);
  for (let i = 0; i < 1920; i += 2) b.writeInt16LE(value, i);
  return b;
}

describe("buildRecorderFfmpegArgs", () => {
  it("re-encodes pcm for mp3/wav and remuxes opus for ogg", () => {
    const mp3 = buildRecorderFfmpegArgs("mp3").join(" ");
    expect(mp3).toContain("libmp3lame");
    expect(mp3).toContain("-f mp3");

    const wav = buildRecorderFfmpegArgs("wav").join(" ");
    expect(wav).toContain("pcm_s16le");
    expect(wav).toContain("-f wav");

    const ogg = buildRecorderFfmpegArgs("ogg").join(" ");
    expect(ogg).toContain("-c:a copy");
    expect(ogg).toContain("-f ogg");
  });
});

describe("ChannelRecorder", () => {
  let dir: string;
  let written: Buffer[];
  let currentTime: number;
  let recorder: ChannelRecorder;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "recorder-test-"));
    written = [];
    currentTime = 1_000_000;
    recorder = new ChannelRecorder(silentLogger, {
      spawnFn: makeFakeSpawn(written),
      now: () => currentTime,
      flushIntervalMs: 60_000, // manual flush via stop() in tests
    });
    stub.decode = () => pcmFrame(1000);
    stub.encode = () => Buffer.from([0xff, 0xee]);
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("writes decoded frames then silence for gaps (wav)", () => {
    recorder.start("wav", join(dir, "out.wav"));
    expect(recorder.isRecording()).toBe(true);

    // Two packets 100ms apart → frames 0 and 5.
    recorder.handleVoicePacket(new Uint8Array([1]));
    currentTime += 100;
    recorder.handleVoicePacket(new Uint8Array([1]));
    currentTime += 200; // stop 200ms later → ~20 frames total

    const handle = recorder.stop();
    expect(handle?.durationMs).toBeGreaterThanOrEqual(300);
    expect(recorder.isRecording()).toBe(false);

    const data = Buffer.concat(written);
    // Frame 0 and 5 carry the decoded PCM value; frames 1-4, 6.. are silence.
    expect(data.length).toBeGreaterThanOrEqual(1920 * 15);
    expect(data.readInt16LE(0)).toBe(1000);
    expect(data.readInt16LE(1920 * 5)).toBe(1000);
    expect(data.readInt16LE(1920 * 2)).toBe(0);
  });

  it("mixes overlapping speakers", () => {
    stub.decode = () => pcmFrame(20000);
    recorder.start("wav", join(dir, "mix.wav"));
    recorder.handleVoicePacket(new Uint8Array([1]));
    recorder.handleVoicePacket(new Uint8Array([2])); // same slot → mixed
    currentTime += 100;
    recorder.stop();
    const data = Buffer.concat(written);
    expect(data.readInt16LE(0)).toBe(20000 + 20000);
  });

  it("start() throws when already recording and stop() is idempotent", () => {
    recorder.start("mp3", join(dir, "a.mp3"));
    expect(() => recorder.start("mp3", join(dir, "b.mp3"))).toThrow(/already/i);
    expect(recorder.stop()).not.toBeNull();
    expect(recorder.stop()).toBeNull();
  });

  it("ogg mode appends raw packets and encoded silence", () => {
    recorder.start("ogg", join(dir, "o.ogg"));
    recorder.handleVoicePacket(new Uint8Array([9]));
    currentTime += 60; // 2 silent frames worth of gap
    recorder.stop();
    const data = Buffer.concat(written);
    expect(data[0]).toBe(9); // raw packet passthrough
    expect(data.length).toBe(1 + 2 + 2); // packet + two silence frames
  });
});

describe("recording files helpers", () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "recorder-files-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  it("records a timestamped, prefixed file name", () => {
    const name = recordingFileName("bot-1", "mp3", new Date(2026, 9, 26, 14, 5, 9));
    expect(name).toBe("bot-1-20261026-140509.mp3");
  });

  it("lists only matching files", () => {
    writeFileSync(join(dir, "bot-1-a.mp3"), "x");
    writeFileSync(join(dir, "bot-1-b.wav"), "xx");
    writeFileSync(join(dir, "other-c.mp3"), "xxx");
    const names = listRecordingFiles(dir, "bot-1").map((f) => f.name);
    expect(names).toContain("bot-1-a.mp3");
    expect(names).toContain("bot-1-b.wav");
    expect(names).not.toContain("other-c.mp3");
  });

  it("sweeps expired recordings only", () => {
    writeFileSync(join(dir, "bot-1-old.mp3"), "x");
    sweepOldRecordings(dir, "bot-1", 1, Date.now() + 2 * 3600_000);
    expect(listRecordingFiles(dir, "bot-1")).toHaveLength(0);
    // retention 0 → keep forever
    writeFileSync(join(dir, "bot-1-keep.mp3"), "x");
    sweepOldRecordings(dir, "bot-1", 0, Date.now() + 10 * 3600_000);
    expect(listRecordingFiles(dir, "bot-1")).toHaveLength(1);
  });

  it("safeRecordingPath rejects traversal", () => {
    expect(typeof safeRecordingPath(dir, "ok.mp3")).toBe("string");
    expect(safeRecordingPath(dir, "../config.json")).toBeNull();
    expect(safeRecordingPath(dir, "a/b.mp3")).toBeNull();
    expect(safeRecordingPath(dir, "..\\x.mp3")).toBeNull();
  });
});
