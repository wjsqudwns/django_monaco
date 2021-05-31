from titanic.models.dataset import Dataset
import pandas as pd


class Service(object):  # manipulating

    dataset = Dataset()

    def new_model(self, payload) -> object:
        this = self.dataset # this변수명 데이터셋 인스턴스를 디스에 넣어줌
        this.context = './data/'
        this.fname = payload
        return pd.read_csv(this.context + this.fname) # 경로와 파일이름을 붙여 읽어옴 리턴값 데이터프레임 타이타닉데이터 자체가 딕셔너리 -> csv = 딕셔너리
        # pd.read_csv -> 딕셔너리로 바로 가져올수있다.

    @staticmethod
    def create_train(this) -> object:
        return this.train.drop('Survived', axis=1)

    @staticmethod
    def create_label(this) -> object:
        return this.train['Survived']

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis=1)
        this.test = this.test.drop([feature], axis=1)
        return this
