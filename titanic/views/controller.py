from titanic.models.dataset import Dataset
from titanic.models.service import Service
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
        this.id = this.test['PassengerId']
        #  불필요한 feature(cabin(객실번호), ticket) 제거해버림 feature -> 편집가능한 컨테이너
        # 타입이 같을경우 여러개 넣어도 된다 (튜플로 변환)
        #   norminal, ordinal 로 정형화

        this = service.embarked_nominal(this)
        this = service.title_norminal(this)
        this = service.gender_norminal(this)  #  노미널해주고 변환
        this = service.age_ordinal(this)
        this = service.fare_ordinal(this)
        this = service.drop_feature(this, 'Fare', 'Age', 'Sex', 'Name', 'Cabin', 'Ticket')

        self.print_this(this)

        return this

    @staticmethod
    def print_this(this):

        print('<type check>')
        print(f'check {type(this.train["Embarked"])}')
        print(f'Train 의 type 은 {type(this.train)} 이다.')
        print(f'Train 의 column 은 {(this.train.columns)} 이다.')
        print(f'Train 의 상위5개행 은\n {(this.train.head(10))} 이다.')
        print(f'4. Train의 null 의 개수\n {this.train.isnull().sum()}개') # null갯수 확인

        print(f'test 의 type 은 {type(this.test)} 이다.')
        print(f'type 의 column 은 {(this.test.columns)} 이다.')
        print(f'Train 의 상위5개행 은 \n{(this.test.head(10))} 이다.')
        print(f'4. Test의 null 의 개수\n {this.test.isnull().sum()}개')

        print('*' * 100)

    def modeling(self, train, test) -> object:
        service = self.service
        this = self.preprocess(train, test)
        this.label = service.create_label(this)
        print(f'----------{type(this.label)}---------')
        this.train = service.create_train(this)
        return this

    def learning(self,this):

        print(f'svc 알고리즘 정확도 {self.service.accracy_by_svm(this)}%')

    def submit(self, train, test):
        this = self.modeling(train, test)
        clf = RandomForestClassifier()
        clf.fit(this.train, this.label)
        prediction = clf.predict(this.test)
        pd.DataFrame({'PassengerId': this.id, 'Survived': prediction}).to_csv('./data/submission.csv', index=False)

#  controller.main() runnable