# DOCUMENT CLASSIFICATION

### Usage

1. **Website Address**
 
    [Link](http://flask-doc.sgn7ieqpcn.us-east-2.elasticbeanstalk.com/)
    
    [Video](https://drive.google.com/file/d/1-wYsYuTTGyr7UJOGbkdsJVPbEtrj8lNv/view?usp=sharing)
    
2. **APIs for query && Test your own dataset**
    
    We provide APIs for users to query a single doc or multiple docs, you can also use them to test your own dataset.
    
   - _**Single query:**_
    
        ~~~~
        Send
        http://flask-doc.sgn7ieqpcn.us-east-2.elasticbeanstalk.com/api/v1.0/predict?words=[your doc]`

        Return
        {
            "Label": "BINDER", 
            "Confidence": 0.3679347894413934
        }
        ~~~~
    
   - **_Multiple query (build your own json file):_**
    
        ~~~~
        Send
        {
                'labels':["labels1","labels2"],
                'words':["words1","words2"]
        }
        
        Return
        {
            "Accuracy": 0.8888888888888888, 
            "Label": ["DELETION OF INTEREST", "RETURNED CHECK", "BILL", "BILL", "BILL", "POLICY CHANGE", "BINDER", "BILL", "CANCELLATION NOTICE"], 
            "Confidence": [0.6, 1.0, 1.0, 0.6, 1.0, 0.85, 0.4, 1.0, 0.6]
        }
        ~~~~

   - Or you can choose to run testAPI.py to test them. (Check the server address and keep it running)

3. **Deployment**

    The service is based on flask and the entrance file is application.py, you can directly run it at your server. The default address is `http://localhost:5000/`
    
    I built it on AWS, you can follow the [official tutorial](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html) to build it on your AWS.
   
    
### Structure

![Home](https://github.com/zzzyyyxxxmmm/doc_clf/blob/master/img/structure.png)

### Introduction
This is a mini-project about document classification: Given you a list of encoded words, you need to output a label. Here I just re-encoded categorical words to integer codes and trained them by different classification algorithms. Then, I used flask to build a server base on python and provided APIs for users to query online. After that I deployed my server at AWS Elastic Beanstalk so that it can be used by anyone. Finally, I also designed an android application for users to query by their mobile phones.

### Classification
This is a common supervised classification problem: we have some features and labels and we need to train them into a model for new inputs. Most of these problems have following steps:
1.	Encode features. Here I used TfidfVectorizer as my encoder and it behaved well comparing to the other two vectorizer.
2.	Split them to train dataset and test dataset
3.	Use different classification algorithm to train the model. Here I used four algorithms to train them.
4.	Use the model to test and score them.

![Home](https://github.com/zzzyyyxxxmmm/doc_clf/blob/master/img/accuracy.png)

![Home](https://github.com/zzzyyyxxxmmm/doc_clf/blob/master/img/cm.png)

### Build Server
I built a website based on python so that users can query any docs conveniently. Flask is a python microframework. By adding some annotation we can simply map the URL to our function and return the HTML page.

I designed three important functions in this website:
1.	Query by text area in the web page. Users can input their query words into the test area and submit them, then the server will load the trained model, classify them and return the result.
2.	Single Query by get request. For a single query, we can add the words as a parameter to the request. It will return a json file.
3.	Muitiple Query by Json. For multiple queries, we send a json file to server and return a json file.

### Deployment
Considering AWS is the most popular cloud server in the world, AWS Elastic Beanstalk is an easy-to-use service for deploying and scaling web applications and services. You can simply upload your code and Elastic Beanstalk automatically handles the deployment, from capacity provisioning, load balancing, auto-scaling to application health monitoring. Just following the official guide , I deployed them to cloud. Before that, I tried Lambada and want to build a cost-effective serverless solution, but it seems I need more time to understand how it works.

### Android Application
Considering I’m not very good at building website or anything about cloud, I designed an android application for this project and hope to show my ability to some extent. 

[Download Address](https://github.com/zzzyyyxxxmmm/doc_clf/blob/master/app-debug.apk)

[Github Address](https://github.com/zzzyyyxxxmmm/docClassifierAndroid)

![Home](https://github.com/zzzyyyxxxmmm/doc_clf/blob/master/img/mobile_structure.png)


Simulator

![Home](https://github.com/zzzyyyxxxmmm/doc_clf/blob/master/img/clf_apk.gif)

Real Phone

![Home](https://github.com/zzzyyyxxxmmm/doc_clf/blob/master/img/real.png)




