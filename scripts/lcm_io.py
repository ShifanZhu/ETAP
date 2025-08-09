import lcm
import threading
import time
from typing import Optional, List, Tuple
from lcm_types.msgs import TrackingCommand, TrackingUpdate

class LcmBridge:
    def __init__(self, cmd_topic: str, upd_topic: str):
        self.lc = lcm.LCM()
        self.cmd_topic = cmd_topic
        self.upd_topic = upd_topic

        self._lock = threading.Lock()
        self._pending_cmd: Optional[TrackingCommand] = None
        self._shutdown = False

        self.lc.subscribe(self.cmd_topic, self._on_cmd)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

        # simple ID allocator for “new” features (optional)
        self._next_id = 10_000_000

    def _on_cmd(self, channel: str, data: bytes):
        msg = TrackingCommand.decode(data)
        with self._lock:
            self._pending_cmd = msg

    def _spin(self):
        while not self._shutdown:
            try:
                self.lc.handle_timeout(50)  # 50 ms
            except Exception:
                pass

    def get_pending_request(self) -> Optional[TrackingCommand]:
        with self._lock:
            cmd = self._pending_cmd
            self._pending_cmd = None
            return cmd

    def publish_update(self, ts_ns: int, ids: List[int], xs: List[float], ys: List[float]):
        msg = TrackingUpdate()
        msg.timestamp_ns = int(ts_ns)
        msg.feature_ids = list(map(int, ids))
        msg.feature_x = list(map(float, xs))
        msg.feature_y = list(map(float, ys))
        self.lc.publish(self.upd_topic, msg.encode())

    def allocate_ids(self, n: int) -> List[int]:
        # If you add support points or detect new tracks, give them stable new IDs
        out = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        return out

    def close(self):
        self._shutdown = True
        self._thread.join(timeout=0.3)
