from titanic.views.controller import Controller
from titanic.templates.plot import Plot

# 이닛 파일은 객체가 아니라서 클래스가 필요없다, 같은 레벨에 이닛에 가져와서 테스트해야함 -> 패키지와 같은 레벨
if __name__ == '__main__':


    while 1:
        menu = input('0:EXIT 1: Data Visualization 2: Modeling 3: Machine leaning 4: Machine release')
        controller = Controller()
        if menu == '0':
            break

        elif menu == '1':
            test = Plot('train.csv')
            #test.draw_survived_dead()
            #test.draw_Pclass()
            #test.draw_sex()
            test.draw_embarked()

        elif menu == '2':

            controller.preprocess('train.csv', 'test.csv')

        elif menu == '3':
            df = controller.modeling('train.csv', 'test.csv')
            controller.learning(df)

        elif menu == '4':
            controller.submit('train.csv', 'test.csv')

        else:
            continue

