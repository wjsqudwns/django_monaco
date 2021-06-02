from titanic.models.dataset import Dataset
from titanic.models.service import Service
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns

rc('font', family=font_manager.FontProperties(fname='C:/Windows/Fonts/H2GTRE.ttf').get_name())  # C:\Windows\Fonts 지원하는 폰트 찾아 쓰기


class Plot(object):
    dataset = Dataset()  # ()는 생성자의 의미 즉 dataset은 인스턴스
    service = Service()  # 공유객체

    #  생성자
    def __init__(self, train):  # 타이타닉의 init -> plot에 init에 값을 넣어줌 -> 서비스 뉴모델메소드에서 반환해서 엔티티 대입 -> 다시 inin에서 test 메소드 호출 값 출력
        self.entity = self.service.new_model(train)  # 서비스에 뉴모델 메소드에 넣어서 반환해주는것 안어렵다 정신차려

        # return이 있다면 type을 명시해 주어야함

    def draw_survived_dead(self):
        this = self.entity
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        this['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
        ax[0].set_title('사망자 vs 생존자')
        ax[0].set_ylabel('')
        ax[1].set_title('사망자 vs 생존자')
        sns.countplot('Survived', data=this, ax=ax[1])
        plt.show()

    #   print(f'Train 의 type 은 {type(this)} 이다.')
    #   print(f'Train 의 column 은 {this.columns} 이다.')
    #   print(f'Train 의 상위 5개 데이터는 {this.head} 이다.')
    #   print(f'Train 의 하위 5개 데이터는 {this.tail} 이다.')

    def draw_Pclass(self):
        this = self.entity  #  카피본으로 사용하겠다. 지금 this는 데이터프레임(엑셀비슷)
        this["Survived"] = this["Survived"].replace(0, "Perish").replace(1, "Survived")
        this["Pclass"] = this["Pclass"].replace(1, "FirstClass").replace(2, "SecondClass").replace(3, "EconomyClass")
        sns.countplot('Pclass', data=this, x='Pclass', hue='Survived')
        plt.show()

    def draw_sex(self):
        this = self.entity
        f, ax = plt.subplots(1, 2, figsize=(18,8))
        this['Survived'][this['Sex'] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True) # ~이면서 할때 ['Survived'][this['Sex'] == 'male']
        this['Survived'][this['Sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)

        ax[0].set_title('남성의 생존비율')
        ax[1].set_title('여성의 생존비율')
        plt.show()

    def draw_embarked(self):
        this = self.entity  # 카피본으로 사용하겠다.
        this["Survived"] = this["Survived"].replace(0, "Perish").replace(1, "Survived")
        this["Embarked"] = this["Embarked"].replace("C", "쉘버그").replace("S", "사우스햄튼").replace("Q", "퀸즈타운")
        sns.countplot(data=this, x='Embarked', hue='Survived')
        plt.show()


# init에서 실행하기 때문에 메인이 나올수 없다. 낱개로 작동하는 코드가 아닌 타이타닉이라는 패키지단위이기 때문이다.

'''Train 의 type 은 <class 'pandas.core.frame.DataFrame'> 이다.
Train 의 column 은 Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object') 이다.
Train 의 상위 5개 데이터는 <bound method NDFrame.head of      PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0              1         0       3  ...   7.2500   NaN         S
1              2         1       1  ...  71.2833   C85         C
2              3         1       3  ...   7.9250   NaN         S
3              4         1       1  ...  53.1000  C123         S
4              5         0       3  ...   8.0500   NaN         S
..           ...       ...     ...  ...      ...   ...       ...
886          887         0       2  ...  13.0000   NaN         S
887          888         1       1  ...  30.0000   B42         S
888          889         0       3  ...  23.4500   NaN         S
889          890         1       1  ...  30.0000  C148         C
890          891         0       3  ...   7.7500   NaN         Q

[891 rows x 12 columns]> 이다.
Train 의 하위 5개 데이터는 <bound method NDFrame.tail of      PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0              1         0       3  ...   7.2500   NaN         S
1              2         1       1  ...  71.2833   C85         C
2              3         1       3  ...   7.9250   NaN         S
3              4         1       1  ...  53.1000  C123         S
4              5         0       3  ...   8.0500   NaN         S
..           ...       ...     ...  ...      ...   ...       ...
886          887         0       2  ...  13.0000   NaN         S
887          888         1       1  ...  30.0000   B42         S
888          889         0       3  ...  23.4500   NaN         S
889          890         1       1  ...  30.0000  C148         C
890          891         0       3  ...   7.7500   NaN         Q

[891 rows x 12 columns]> 이다.
0:EXIT 1: Data Visualization 2: Modeling 3: Machine leaning 4: Machine release'''
