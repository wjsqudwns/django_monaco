from titanic.models.dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class Service(object):  # manipulating

    dataset = Dataset()

    def new_model(self, payload) -> object:
        this = self.dataset # this변수명 데이터셋 인스턴스를 디스에 넣어줌
        this.context = './data/'
        this.fname = payload

        return pd.read_csv(this.context + this.fname) # 경로와 파일이름을 붙여 읽어옴/
        # 리턴값 데이터프레임/타이타닉데이터 자체가 딕셔너리 -> c
        # pd.read_csv -> 딕셔너리로 바로 가져올수있다.csv = 딕셔너리

    @staticmethod
    def create_train(this) -> object:
        return this.train.drop('Survived', axis=1)

    @staticmethod
    def create_label(this) -> object:
        return this.train['Survived']

    @staticmethod
    def drop_feature(this, *feature) -> object:  #  키 값을 반복시켜주어야한다 두개값이 들어오기 때문이다.
        for i in feature:
            this.train = this.train.drop([i], axis=1)  # axis=1 -> column 세로축을 지정 0-> 가로축을 지워라
            this.test = this.test.drop([i], axis=1)
        return this

    @staticmethod
    def embarked_nominal(this) -> object:  # 데이터프레임 객체
        this.train = this.train.fillna({'Embarked': 'S'})  # 데이터 프레임의 딕셔너리에 비어있는 값을 채워준다 그래서 na오류가 안나게 한다.
        # S라고 준 이유는 다른 항구에 비해 s 항구는 꼼꼼하게 승선을 확인하지 않앗고 비율또한 많기 때문에 S로한다 (오차는 날수밖에없음)
        # Embarked항목만 빈곳을 S로 채운다
        this.test = this.test.fillna({'Embarked': 'S'})
        this.train['Embarked'] = this.train['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}) # 앞에 문자를 뒤에 숫자로 치환
        this.test['Embarked'] = this.test['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
        #  항상 트렌인과 테스트는 같이 적용시켜주어야한다.
        # this.train['Embarked'] 는 시리즈 타입으로 반환이 되는데 시리즈는 리스트처럼 처리할 수 있다
        return this

    @staticmethod
    def fare_ordinal(this) -> object:
        this.test['Fare'] = this.test['Fare'].fillna(1)
        train = this.train
        test = this.test
        labels = {1, 2, 3, 4}
        #  밑에 코드는 list 를 2개 주고 안에서 딕셔너리될거라는것을 알앗고 이거는 하나만 주기때문에 딕셔너리로 넣어야함
        bins = [-1, 8, 15, 31, np.inf]

        for these in train, test:
            these['FareBand'] = pd.cut(these['Fare'], bins=bins, labels=labels)

        #this.train['FareBand'] = pd.qcut(this.train['Fare'], 4)
        #this.test['FareBand'] = pd.qcut(this.test['Fare'], 4)
        #q컷을 통해 값을 알아냄, bins 사용으로 반복문 사용가능하게함 원래는 test와 train의 값 범위가 달랏으니 bins로 같게만들어줌

        return this


    @staticmethod
    def title_norminal(this) -> object:
        combine = [this.train, this.test]

        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False) #  추출한것 빼고 나머지는 버리기위해 라벨로출력

        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            dataset['Title'] = dataset['Title'].replace('Mlle', 'Mr')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Rare') # 앞에문자를 통합하여 뒤에문자로 치환
            # title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}

            dataset['Title'] = dataset['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6})
            dataset['Title'] = dataset['Title'].fillna(0)

        return this

            #  \.  이스케이프 문자 . 앞에있는 문자 추출 Ms. <- 점

            #  Column 추가 하는것 df[' ']

            #  정규표현식[A-Za-z] / 문자열클래스[] 글자패턴()  A-Z 알파벳 대문자 a-z 알파벳소문자 -> 숫자가 있으면 안됨/ + 값이 반드시 있어야 한다 알파벳을 반드시 하나이상을 포함해야한다



    @staticmethod
    def gender_norminal(this) -> object:
        combine = [this.train, this.test] # this.train = combine[0] this.test = combine[1] 컴바인이 자동으로 먹


        gender_mapping = {'male': 0, 'female': 1}

        for i in combine:
            i['Gender'] = i['Sex'].map(gender_mapping)

        this.train = combine[0]
        this.test = combine[1]

        return this

    @staticmethod
    def age_ordinal(this) -> object:
        train = this.train
        test = this.test

        train['Age'] = train['Age'].fillna(-0.5) #  측정할 수는 없지만 null값이면 오류가 나기때문에 바꿔주고 나중에 라벨로 묶어 처리ㅏㄴ다
        test['Age'] = test['Age'].fillna(-0.5)

        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]  #  최대 최소값 정해 그룹별로 나누어 묶어준다
        #  -1 은 미상 측정할수없는 값으로 하고(null을 0으로 주엇기때문) 구간별로 값을 따로 준 이유는 유아 청소녀 노년을 따로 구분하기 위해서이다.
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
        age_title_mapping = {'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5,
                             'Adult': 6, 'Senior': 7}

        for i in train, test:
            i['AgeGroup'] = pd.cut(i['Age'], bins=bins, labels=labels)  # new feature bins 와 label을 묶어준다 컷 메소드의 labels 는 키값
            # list 2개 -> 딕셔너리
            i['AgeGroup'] = i['AgeGroup'].map(age_title_mapping)

        return this

    @staticmethod
    def create_k_fold() -> object:
        return KFold(n_splits=10, shuffle=True, random_state=0)

    def accracy_by_svm(self, this):
        score = cross_val_score(SVC(), this.train, this.label, cv=KFold(n_splits=10, shuffle=True, random_state=0), n_jobs=1, scoring='accuracy')
        return round(np.mean(score) * 100, 2)



# 요금을 구간별로 구분


# fare_band_norminal
# title_norminal
# gender_norminal
# age_ordinal
# create_k_fold