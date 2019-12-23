import _CXX_i8ie


class Tensor:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return ((self.numpy() - self.zero_point) * self.scale).__repr__()

    def __eq__(self, obj):
        return _CXX_i8ie.tensor(self.numpy() == obj.numpy())

    def reshape(self, *args):
        return Tensor(self.data.reshape(list(args)))

    def numpy(self):
        return self.data.numpy()

    def sum(self):
        return self.numpy().sum()

    @property
    def shape(self):
        return self.data.numpy().shape

    @property
    def scale(self):
        return self.data.scale()

    @property
    def zero_point(self):
        return self.data.zero_point()

    @property
    def dtype(self):
        pass
