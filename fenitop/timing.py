
import time
from collections import defaultdict

class TimingStats:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimingStats, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance
    
    def reset(self):
        self.timers = defaultdict(float)
        self.counts = defaultdict(int)
        self.current_starts = {}
        
    def start(self, name):
        self.current_starts[name] = time.perf_counter()
        
    def stop(self, name):
        if name in self.current_starts:
            elapsed = time.perf_counter() - self.current_starts[name]
            self.timers[name] += elapsed
            self.counts[name] += 1
            del self.current_starts[name]
            return elapsed
        return 0.0
        
    def get_summary(self):
        return dict(self.timers)

# Global instance
stats = TimingStats()
