// Types des messages échangés avec le serveur (voir server/protocol.py).

export type Point = [number, number];

export interface TrackMessage {
  type: "track";
  seed: number;
  width: number;
  length: number;
  center: Point[];
  left: Point[];
  right: Point[];
  ray_angles: number[];
  ray_max_range: number;
}

export interface CarState {
  x: number;
  y: number;
  heading: number;
  speed: number;
  yaw_rate: number;
}

export type Outcome = "running" | "collision" | "finished" | "timeout";

export type Mode = "manual" | "pilot";

export interface StateMessage {
  type: "state";
  mode: Mode;
  step: number;
  car: CarState;
  rays: number[];
  reward: number;
  completion: number;
  outcome: Outcome;
  done: boolean;
}

export type View = "play" | "replay";

export interface ViewMessage {
  type: "view";
  view: View;
}

export interface RunInfo {
  name: string;
  generations: number;
}

export interface RunsMessage {
  type: "runs";
  runs: RunInfo[];
  current: string | null;
}

export interface HistoryRow {
  generation: number;
  best: number;
  mean: number;
  val_fitness: number | null;
  val_finished: number | null;
  val_lap_s: number | null;
}

export interface HistoryMessage {
  type: "history";
  run: string | null;
  rows: HistoryRow[];
}

export type CarStatus = "running" | "finished" | "crashed" | "stalled" | "timeout";

export interface ReplayCar {
  x: number;
  y: number;
  heading: number;
  status: CarStatus;
  completion: number;
}

export interface ReplayStateMessage {
  type: "replay_state";
  run: string | null;
  generation: number;
  n_generations: number;
  circuit: number;
  n_circuits: number;
  speed: number;
  step: number;
  running: number;
  fitness: number[];
  cars: ReplayCar[];
}

export type ServerMessage =
  | TrackMessage
  | StateMessage
  | ViewMessage
  | RunsMessage
  | HistoryMessage
  | ReplayStateMessage;

export interface ResetMessage {
  type: "reset";
  seed?: number; // absent : le serveur tire une seed au hasard
}

export interface ModeMessage {
  type: "mode";
  mode: Mode;
}

export interface ViewSelectMessage {
  type: "view";
  view: View;
}

// Champs absents : le serveur garde la valeur actuelle.
export interface ReplaySelectMessage {
  type: "replay";
  run?: string;
  generation?: number;
  circuit?: number;
  speed?: number;
}

export type ClientMessage =
  | InputMessage
  | ResetMessage
  | ModeMessage
  | ViewSelectMessage
  | ReplaySelectMessage;

export interface InputMessage {
  type: "input";
  steering: number;
  throttle: number;
}
