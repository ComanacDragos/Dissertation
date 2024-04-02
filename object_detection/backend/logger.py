class Logger:
    @staticmethod
    def log(msg):
        print(msg)

    def __call__(self, msg):
        self.log(msg)


logger = Logger()