
import datetime
import pulsar
import os

os.environ['CLASSPATH'] = "./consumerBanking-1.0-SNAPSHOT-jar-with-dependencies.jar"

from jnius import autoclass, JavaException


def parserInLongMsg(inlongBytes):
    try:
        InLongMsg = autoclass('com.companyname.bank.App')
        ls = InLongMsg.parserInLongMsg(inlongBytes)
        ls = [bytearray(i.tolist()).decode('utf-8') for i in ls]
        return ls
    except JavaException as ex:
        print(ex.stacktrace)
        return []


client = pulsar.Client('pulsar://localhost:6650')


def consumerCreator(sub, consumerName, fileType):
    global client

    consumer = client.subscribe(sub, consumerName)
    file = sub[sub.rindex('/') + 1:]

    if not os.path.exists(file):
        os.makedirs(file)

    fileName = '{}/{}_{}.{}'.format(file, file, datetime.datetime.now().strftime('%Y%m%d%H%M%S'), fileType.lower())

    while True:
        msg = consumer.receive()
        try:
            data = msg.data()
            dataToLine = parserInLongMsg(data)
            with open(fileName, 'a') as f:
                for line in dataToLine:
                    f.write(line)
                    f.write('\n')

            print('catch data {} at {} by {}'.format(len(dataToLine), msg.publish_timestamp(), consumerName))
            consumer.acknowledge(msg)
        except Exception as e:
            print(e)
            # Message failed to be processed
            consumer.negative_acknowledge(msg)

    client.close()


if __name__ == '__main__':
    # consumerCreator('persistent://public/personal_loan_default_forecast/test_public', 'con_test_public', 'csv')
    # consumerCreator('persistent://public/personal_loan_default_forecast/train_public', 'con_train_public', 'csv')
    consumerCreator('persistent://public/personal_loan_default_forecast/train_internet', 'con_train_internet', 'csv')
