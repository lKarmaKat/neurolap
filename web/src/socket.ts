import type { ClientMessage, ServerMessage, StateMessage, TrackMessage } from "./protocol";

// État reçu du serveur : la boucle de dessin le lit, le réseau l'écrit.
export interface GameData {
  track: TrackMessage | null;
  state: StateMessage | null;
  connected: boolean;
}

export interface Connection {
  data: GameData;
  send(msg: ClientMessage): void;
}

// `onOpen` : appelé à chaque (re)connexion, pour renvoyer l'entrée courante.
export function connect(url: string, onOpen?: () => void): Connection {
  const data: GameData = { track: null, state: null, connected: false };
  let ws: WebSocket | null = null;

  const open = (): void => {
    const sock = new WebSocket(url);
    ws = sock;
    sock.onopen = () => {
      data.connected = true;
      onOpen?.();
    };
    sock.onmessage = (ev: MessageEvent<string>) => {
      const msg = JSON.parse(ev.data) as ServerMessage;
      if (msg.type === "track") data.track = msg;
      else if (msg.type === "state") data.state = msg;
    };
    sock.onclose = () => {
      data.connected = false;
      setTimeout(open, 1000); // reconnexion automatique
    };
  };
  open();

  return {
    data,
    send(msg) {
      if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
    },
  };
}
