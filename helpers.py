from sklearn.preprocessing import normalize
from components import MappingModule

class LoadImages(MappingModule):
    def process(self, document):
        document.load_img_pil()



class EncToNone(MappingModule):
    def process(self, document):
        document.encoding = None

class DescToEnc(MappingModule):
    def process(self, document):
        document.encoding = document.descriptors

class DescToNone(MappingModule):
    def process(self, document):
        document.descriptors = None
        
def clean_descriptors(a):
    return DescToNone()(a)
        


class L2Norm(MappingModule):
    def __init__(self, attr):
        self.attr = attr

    def process(self, document):
        at = getattr(document, self.attr)
        setattr(document, self.attr, normalize(at, norm='l2', axis=1))            

