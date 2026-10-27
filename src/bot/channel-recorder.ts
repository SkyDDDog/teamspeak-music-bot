import { spawn, type ChildProcess } from "node:child_process";
import { mkdirSync, readdirSync, statSync, unlinkSync } from "node:fs";
import { basename, isAbsolute, join, resolve } from "node:path";
import opusModule from "@discordjs/opus";
import { getFfmpegCommand } from "../audio/player.js";
import type { Logger } from "../logger.js";
import type { RecordingFormat } from "../data/config.js";

const { OpusEncoder } = opusModule;

const SAMPLE_RATE = 48000;
const FRAME_DURATION_MS = 20;
/** Mono frame: 48kHz * 20ms = 960 samples, 16-bit. */
const MONO_FRAME_SAMPLES = (SAMPLE_RATE * FRAME_DURATION_MS) / 1000;
const MONO_FRAME_BYTES = MONO_FRAME_SAMPLES * 2;

export interface RecordedFile {
  name: string;
  size: number;
  createdAt: number; // epoch ms
}

export interface RecordingHandle {
  file: string;
  format: RecordingFormat;
  durationMs: number;
}

/** Injected process spawner so tests can fake the ffmpeg child process. */
export type SpawnFn = (
  cmd: string,
  args: string[],
) => Pick<ChildProcess, "stdin" | "kill" | "unref" | "on" | "once"> & { pid?: number };

export interface ChannelRecorderOptions {
  /** Test seams. */
  spawnFn?: SpawnFn;
  now?: () => number;
  /** Flush cadence for completed frames (default 250ms). */
  flushIntervalMs?: number;
}

/** ffmpeg output args per recording format. */
export function buildRecorderFfmpegArgs(format: RecordingFormat): string[] {
  // Raw mono 48kHz s16le (or opus packets for ogg) on stdin → container on disk.
  if (format === "ogg") {
    // Packets are already Opus (mono 48k); remux without re-encoding.
    return [
      "-f", "opus",
      "-ar", String(SAMPLE_RATE),
      "-ac", "1",
      "-i", "pipe:0",
      "-c:a", "copy",
      "-f", "ogg",
    ];
  }
  const codec = format === "mp3" ? "libmp3lame" : "pcm_s16le";
  const muxer = format === "mp3" ? "mp3" : "wav";
  return [
    "-f", "s16le",
    "-ar", String(SAMPLE_RATE),
    "-ac", "1",
    "-i", "pipe:0",
    "-c:a", codec,
    "-b:a", "96k",
    "-f", muxer,
  ];
}

/**
 * Records the speech of OTHER clients in the channel (raw Opus packets from
 * `TS3VoiceActivity.data`) to MP3/WAV/OGG via ffmpeg.
 *
 * TeamSpeak exposes voice packets but no "stopped talking" event, so the
 * timeline is reconstructed on 20ms frame slots indexed by arrival time:
 * every incoming packet is decoded and mixed into its wall-clock frame slot,
 * and a periodic flush writes consecutive completed slots to ffmpeg, filling
 * silent gaps so the recording stays continuous. OGG mode skips decoding and
 * writes the Opus packets directly (silence gaps are re-encoded Opus).
 */
export class ChannelRecorder {
  private ffmpeg: ReturnType<SpawnFn> | null = null;
  private decoder: { decode(b: Buffer): Buffer } | null = null;
  private silenceEncoder: { encode(b: Buffer): Buffer } | null = null;
  private startTime = 0;
  private framesWritten = 0;
  /** frameIndex -> mixed mono PCM (or Opus packet in ogg mode). */
  private pending = new Map<number, Buffer>();
  private flushTimer: ReturnType<typeof setInterval> | null = null;
  private outFile = "";
  private format: RecordingFormat = "mp3";
  private readonly spawnFn: SpawnFn;
  private readonly now: () => number;
  private readonly flushIntervalMs: number;

  constructor(private readonly logger: Logger, options: ChannelRecorderOptions = {}) {
    this.spawnFn =
      options.spawnFn ??
      ((cmd, args) =>
        spawn(cmd, args, { stdio: ["pipe", "ignore", "ignore"] }) as unknown as ReturnType<SpawnFn>);
    this.now = options.now ?? (() => Date.now());
    this.flushIntervalMs = options.flushIntervalMs ?? 250;
  }

  isRecording(): boolean {
    return this.ffmpeg !== null;
  }

  currentFile(): string {
    return this.outFile;
  }

  /** Wall-clock duration of audio committed so far. */
  getDurationMs(): number {
    return this.framesWritten * FRAME_DURATION_MS;
  }

  start(format: RecordingFormat, outFile: string): void {
    if (this.ffmpeg) throw new Error("Recording already in progress");

    mkdirSync(resolve(dirnameOf(outFile)), { recursive: true });
    this.format = format;
    this.outFile = outFile;
    this.startTime = this.now();
    this.framesWritten = 0;
    this.pending.clear();

    if (format !== "ogg") {
      const dec = new OpusEncoder(SAMPLE_RATE, 1);
      this.decoder = { decode: (b) => dec.decode(b) };
    } else {
      const enc = new OpusEncoder(SAMPLE_RATE, 1);
      this.silenceEncoder = { encode: (b) => enc.encode(b) };
      this.decoder = null;
    }

    this.ffmpeg = this.spawnFn(getFfmpegCommand(), [...buildRecorderFfmpegArgs(format), outFile]);
    this.ffmpeg.on("error", (err) => {
      this.logger.error({ err, file: outFile }, "Recording ffmpeg error");
      this.teardown();
    });
    this.ffmpeg.unref?.();

    this.flushTimer = setInterval(() => this.flush(), this.flushIntervalMs);
    this.flushTimer.unref?.();
  }

  /**
   * Feed one voice packet from another client. Mixed into the current 20ms
   * frame slot(s); opus packets may decode to 20/40/60ms.
   */
  handleVoicePacket(data: Uint8Array): void {
    if (!this.ffmpeg || !this.stdinOpen()) return;

    const elapsed = this.now() - this.startTime;
    const frameIndex = Math.max(0, Math.floor(elapsed / FRAME_DURATION_MS));

    if (this.format === "ogg") {
      // Raw packets: append as-is. TS voice frames are self-contained 20ms+
      // Opus packets, so sequential append (gap-filled at flush) is correct.
      this.mixInto(frameIndex, Buffer.from(data), false);
      return;
    }

    let pcm: Buffer;
    try {
      pcm = this.decoder!.decode(Buffer.from(data));
    } catch (err) {
      this.logger.debug({ err }, "Skipping undecodable voice packet");
      return;
    }
    // Split multi-frame decodes (40/60ms) across consecutive slots.
    for (let off = 0; off < pcm.length; off += MONO_FRAME_BYTES) {
      this.mixInto(frameIndex + off / MONO_FRAME_BYTES, pcm.subarray(off, off + MONO_FRAME_BYTES), true);
    }
  }

  /** Mix (PCM) or append (OGG packet) a chunk into a frame slot. */
  private mixInto(frameIndex: number, chunk: Buffer, isPcm: boolean): void {
    const existing = this.pending.get(frameIndex);
    if (!existing || !isPcm) {
      this.pending.set(frameIndex, chunk);
      return;
    }
    const out = Buffer.from(existing); // copy: may be a shared subarray
    const n = Math.min(out.length, chunk.length);
    for (let i = 0; i < n; i += 2) {
      const sum = out.readInt16LE(i) + chunk.readInt16LE(i);
      out.writeInt16LE(Math.max(-32768, Math.min(32767, sum)), i);
    }
    this.pending.set(frameIndex, out);
  }

  /**
   * Write every completed frame (all slots strictly before the current one,
   * plus one slot of latency headroom) to ffmpeg, emitting encoded silence for
   * slots nobody spoke into.
   */
  private flush(): void {
    if (!this.ffmpeg || !this.stdinOpen()) return;
    const currentFrame = Math.floor((this.now() - this.startTime) / FRAME_DURATION_MS);
    // Keep the in-progress slot pending; flush everything before it minus one
    // slot so a straggler packet for a just-finished slot can still land.
    const flushUpTo = currentFrame - 1;
    while (this.framesWritten < flushUpTo) {
      const idx = this.framesWritten;
      const chunk = this.pending.get(idx);
      if (chunk) {
        this.pending.delete(idx);
        this.writeStdin(chunk);
      } else if (this.format === "ogg") {
        this.writeStdin(this.silenceEncoder!.encode(Buffer.alloc(MONO_FRAME_BYTES)));
      } else {
        this.writeStdin(SILENCE);
      }
      this.framesWritten++;
    }
  }

  private writeStdin(chunk: Buffer): void {
    try {
      this.ffmpeg!.stdin!.write(chunk);
    } catch (err) {
      this.logger.warn({ err }, "Recording stdin write failed");
    }
  }

  private stdinOpen(): boolean {
    return !!this.ffmpeg?.stdin && !this.ffmpeg.stdin.destroyed && this.ffmpeg.stdin.writable;
  }

  /** Final flush then close ffmpeg stdin so it finalizes the container. */
  stop(): RecordingHandle | null {
    if (!this.ffmpeg) return null;
    this.flush();
    const handle: RecordingHandle = {
      file: this.outFile,
      format: this.format,
      durationMs: this.getDurationMs(),
    };
    try {
      this.ffmpeg.stdin!.end();
    } catch {
      /* already closed */
    }
    this.teardown();
    this.logger.info({ ...handle }, "Recording finished");
    return handle;
  }

  private teardown(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    if (this.ffmpeg) {
      try {
        this.ffmpeg.kill();
      } catch {
        /* already gone */
      }
      this.ffmpeg = null;
    }
    this.decoder = null;
    this.silenceEncoder = null;
    this.pending.clear();
    this.outFile = "";
    this.startTime = 0;
    this.framesWritten = 0;
  }

  dispose(): void {
    if (this.ffmpeg) this.stop();
  }
}

const SILENCE = Buffer.alloc(MONO_FRAME_BYTES);

function dirnameOf(p: string): string {
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return i === -1 ? "." : p.slice(0, i);
}

export interface RecordingDirContext {
  dir: string;
  prefix: string; // per-bot filename prefix, e.g. "bot-1"
}

export function recordingFileName(prefix: string, format: RecordingFormat, now = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp =
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  return `${prefix}-${stamp}.${format}`;
}

/** List recording files for a bot prefix, newest first. */
export function listRecordingFiles(dir: string, prefix: string): RecordedFile[] {
  try {
    return readdirSync(dir)
      .filter((f) => f.startsWith(`${prefix}-`) && /\.(mp3|wav|ogg)$/.test(f))
      .map((name) => {
        const st = statSync(join(dir, name));
        return { name, size: st.size, createdAt: st.mtimeMs };
      })
      .sort((a, b) => b.createdAt - a.createdAt);
  } catch {
    return [];
  }
}

/** Delete recordings older than maxRetentionHours (0 = keep forever). */
export function sweepOldRecordings(dir: string, prefix: string, maxRetentionHours: number, now = Date.now()): number {
  if (!(maxRetentionHours > 0)) return 0;
  let deleted = 0;
  for (const f of listRecordingFiles(dir, prefix)) {
    if (now - f.createdAt > maxRetentionHours * 3600_000) {
      try {
        unlinkSync(join(dir, f.name));
        deleted++;
      } catch {
        /* best-effort */
      }
    }
  }
  return deleted;
}

/** Resolve a download request safely: reject path traversal. */
export function safeRecordingPath(dir: string, fileName: string): string | null {
  if (!fileName || fileName.includes("/") || fileName.includes("\\") || fileName.includes("..")) {
    return null;
  }
  const full = resolve(join(dir, fileName));
  const root = resolve(isAbsolute(dir) ? dir : join(process.cwd(), dir));
  return full.startsWith(root + "\\") || full.startsWith(root + "/") ? full : null;
}

export { basename };
