import json
import requests
import preProcess


dataPath="dataset/shuffled-full-set-hashed.csv"
test_num=1000
url_path='http://flask-env.8x6jvjnvrc.us-east-2.elasticbeanstalk.com/'

class Test:

    def __init__(self):
        self.p = preProcess.PreProcess(test_num, dataPath)

    def testMul(self):
        # curl -X POST -H "application/json" -d '{"words":"6666"}' http://localhost:5000/api/v1.0/predictmore
        self.p.run()
        d = {}
        d['labels'] = self.p.labels[:test_num-1]
        d['words'] = self.p.words[:test_num-1]
        data_json = json.dumps(d)
        r = requests.post(url_path+'api/v1.0/predictmore', data=data_json)
        print(r.text)


    def testSingle(self):
        r = requests.get(url_path+'api/v1.0/predict?words=666')
        print(r.text)



if __name__ == '__main__':
    t=Test()
    t.testSingle()
    t.testMul()