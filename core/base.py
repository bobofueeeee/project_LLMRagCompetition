from abc import ABC, abstractmethod

class question(ABC):
    @abstractmethod
    def txt_read(self,path)->list:
        pass

    @abstractmethod
    def pdf_read(self,path)->list:
        pass

    @abstractmethod
    def json_read(self,path)->list:
        pass

class word(ABC):
    @abstractmethod
    def word2vec(self,word):
        pass

    @abstractmethod
    def word2sql(self):
        pass

class dbbase(ABC):
    @abstractmethod
    def sqllite_read(self):
        pass

    @abstractmethod
    def vecdbbase_read(self):
        pass

    def vecdbbase_save(self):
        pass

class llmmodel(ABC):
    @abstractmethod
    def llmodel_loader(self):
        pass

    def llmodel_input(self):
        pass

    def llmodel_output(self):
        pass






