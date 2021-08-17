import os
import time

class mz_logger:
    def __init__(self, out_path, out_log = 'out_log.log', remove=False):
        self.out_log = out_path + out_log

        os.makedirs(out_path, exist_ok=True)
        if os.path.exists(self.out_log) and remove:
            os.remove(self.out_log)
        
    def printlog(self, log):
        print('|'+log, file=open(self.out_log, 'a+'))

    def newline(self):
        print('', file=open(self.out_log, 'a+'))

    def printlog_conti(self, log):
        print('|'+log, end='', file=open(self.out_log, 'a+'))

    def timer_start(self, log=False):
        self.t_start = time.time()
        if log:
            print('|timer start', file=open(self.out_log, 'a+'))

    def timer_end(self):
        self.timer = time.time() - self.t_start
        print('|time: %f'%self.timer, file=open(self.out_log, 'a+'))

    def recreate(self):
        if os.path.exists(self.out_log):
            os.remove(self.out_log)


