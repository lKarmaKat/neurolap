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

export interface StateMessage {
  type: "state";
  step: number;
  car: CarState;
  rays: number[];
  reward: number;
  completion: number;
  outcome: Outcome;
  done: boolean;
}

export type ServerMessage = TrackMessage | StateMessage;

export interface ResetMessage {
  type: "reset";
  seed?: number; // absent : le serveur tire une seed au hasard
}

export type ClientMessage = InputMessage | ResetMessage;

export interface InputMessage {
  type: "input";
  steering: number;
  throttle: number;
}
