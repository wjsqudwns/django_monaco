from titanic.models.dataset import Dataset
import pandas as pd


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
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis=1) # axis=1 -> column 세로축을 지정 0-> 가로축을 지워라
        this.test = this.test.drop([feature], axis=1)
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
    def fare_band_fill_na(this) -> object:

        return this

    @staticmethod
    def title_norminal(this) -> object:
        combine = [this.train, this.test]

        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False) #  추출한것 빼고 나머지는 버리기위해

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

        return this

    @staticmethod
    def create_k_fold(this) -> object:

        return this

# 요금을 구간별로 구분


# fare_band_norminal
# title_norminal
# gender_norminal
# age_ordinal
# create_k_fold