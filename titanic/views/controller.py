from titanic.models.dataset import Dataset
from titanic.models.service import Service
import pandas as pd


class Controller(object):

    dataset = Dataset()
    service = Service()

    # 원 데이터 값을 바꿔야 하기 때문에 리턴을주고 init을 안쓴다

    def preprocess(self, train, test) -> object:  # 전처리작업
        service = self.service
        this = self.dataset
        this.train = service.new_model(train)
        this.test = service.new_model(test)

        print(f'Train 의 type 은 {type(this.train)} 이다.')
        print(f'Train 의 column 은 {(this.train.columns)} 이다.')
        print(f'test 의 type 은 {type(this.test)} 이다.')
        print(f'type 의 column 은 {(this.test.columns)} 이다.')
        return this