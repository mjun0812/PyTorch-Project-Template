class BaseTransform:
    def __call__(self, data: dict) -> dict:
        return data

    def __repr__(self):
        args = [f"{k}={v}" for k, v in self.__dict__.items()]
        if len(args) == 0:
            return self.__class__.__name__ + "()"
        return self.__class__.__name__ + "(" + ", ".join(args) + ")"
