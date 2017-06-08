
class FindExtension:

    def __init__(self, filename='string.extension'):
        self.filename = filename
        self.filenamereversed = filename[::-1]
        index = 0
        for i in self.filenamereversed:
            index += 1
            if i == '.':
                break
            self.dotindex = -1 - index

    def remove_extension(self):
        return self.filename[:self.dotindex]

    def rename_extension(self, new_extension='.string'):
        return self.filename[:self.dotindex]+new_extension

    def add_name_before_extension(self, new_name='string'):
        return self.filename[:self.dotindex]+new_name+self.filename[self.dotindex:]

#
# a = FindExtension('filename.jpg')
# print(a.filenamereversed)
# print(a.dotindex)
# name = 'filename.jpg'
#
# print(name[a.dotindex])
#
# print(a.remove_extension())
#
#
# print(a.rename_extension('.png'))
#
# print(a.add_name_before_extension('_grey'))