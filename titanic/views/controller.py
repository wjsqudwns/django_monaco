from titanic.models.dataset import Dataset
from titanic.models.service import Service
import pandas as pd


class Controller(object):

    dataset = Dataset()
    service = Service()

    # 원 데이터 값을 바꿔야 하기 때문에 리턴을주고 init을 안쓴다

    def preprocess(self, train, test) -> object:  # 전처리작업,
        service = self.service
        this = self.dataset

        #  초기 모델 생성
        this.train = service.new_model(train)
        this.test = service.new_model(test)

        #  불필요한 feature(cabin(객실번호), ticket) 제거해버림 feature -> 편집가능한 컨테이너
        this = service.drop_feature(this, 'Ticket')
        this = service.drop_feature(this, 'Cabin')

        #   norminal, ordinal 로 정형화
        this = service.embarked_nominal(this)

        this = service.title_norminal(this)
        this = service.drop_feature(this, 'Name')


        this = service.gender_norminal(this)  #  노미널해주고 변환
        this = service.drop_feature(this, 'Sex')

        self.print_this(this)
        return this

    @staticmethod
    def print_this(this):

        print('<type check>')
        print(f'check {type(this.train["Embarked"])}')
        print(f'Train 의 type 은 {type(this.train)} 이다.')
        print(f'Train 의 column 은 {(this.train.columns)} 이다.')
        print(f'Train 의 상위5개행 은 {(this.train.head())} 이다.')

        print('*')

        print(f'test 의 type 은 {type(this.test)} 이다.')
        print(f'type 의 column 은 {(this.test.columns)} 이다.')
        print(f'Train 의 상위5개행 은 {(this.test.head())} 이다.')

