import datetime
import json

import boto3
import requests

#from botocore.exceptions import ClientError

from env import USER_STORAGE_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

def load_bd():
    dynamodb = boto3.resource(
        'dynamodb',
        # endpoint_url=os.environ.get('USER_STORAGE_URL'),
        endpoint_url=USER_STORAGE_URL,
        region_name='ru-central1',
        # aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        # aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    return  dynamodb

def read_user(user_id, dynamodb=None):
    if not dynamodb:
        dynamodb = load_bd()
    table = dynamodb.Table('Users')
    try:
        response = table.get_item(Key={'user_id': str(user_id)})

        if 'Item' in response.keys():
            return True
        else:
            return False
    except:
        #print(e.response['Error']['Message'])
        print("error read")
        pass
    #else:
    #    return

    return None

def create_user(user_id, date, dynamodb=None):
    if not dynamodb:
        dynamodb = load_bd()

    table = dynamodb.Table('Users')

    date = datetime.datetime.now()
    response = table.put_item(
        Item={
        'user_id': str(user_id),
        'date': str(date),
        'count': str(0),
        'count_curr_day': str(0)
        }
    )
    return response

def update_user(user_id, date, dynamodb=None):
    if not dynamodb:
        dynamodb = load_bd()

    table = dynamodb.Table('Users')

    response = table.get_item(Key={'user_id': str(user_id)})

    date = datetime.datetime.now()
    date_last = response['Item']['date']
    if str(date).split()[0] == date_last.split()[0]:
        count_curr_day = int(response['Item']['count_curr_day']) + 1
    else:
        count_curr_day = 0


    response = table.update_item(
        Key={'user_id': str(user_id)},
        UpdateExpression="set date=:r, count=:p, count_curr_day=:c",
        ExpressionAttributeValues={
            ':r': str(date), ':p': str(int(response['Item']['count']) + 1), ':c':str(count_curr_day)}
    )
    return response


def setlanguage(user_id, language = 'eng'):
    dynamodb = load_bd()

    table = dynamodb.Table('Language')

    response = table.get_item(Key={'user_id': str(user_id)})

    if 'Item' in response.keys():
        response = table.update_item(
            Key={'user_id': str(user_id)},

            UpdateExpression="set language=:l",
            ExpressionAttributeValues={
                ':l': language}
        )
    else:
        response = table.put_item(
            Item={
                'user_id': str(user_id),
                'language': language,
            }
        )
    return response



def get_language(user_id):
    dynamodb = load_bd()
    table = dynamodb.Table('Language')

    response = table.get_item(Key={'user_id': str(user_id)})
    if 'Item' in response.keys():
        return response['Item']['language']
    else:
        #return None
        return 'eng'


def check_advertisement(ad_id, language):
    dynamodb = load_bd()
    table = dynamodb.Table('Advertisement')

    response = table.get_item(Key={'ad_id': str(ad_id)})
    if response['Item']['active'] == '1' and response['Item']['language'] == language:

        return True, response['Item']['type']
    else:
        return False, None




def get_iam_token_from_oath(oauth_token = 'y0_AgAAAAAJsguRAATuwQAAAADdJXSo3oLNM7_TThK04dqPw2L_64BvI8A', iam_url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'):
    response = requests.post(iam_url, json={"yandexPassportOauthToken": oauth_token})
    json_data = json.loads(response.text)
    if json_data is not None and 'iamToken' in json_data:
        return json_data['iamToken']
    return None

def get_update_iam_token():
    dynamodb = load_bd()

    table = dynamodb.Table('Tokens')
    response = table.get_item(Key={'token_name': str('iam_token')})

    date_reg = response['Item']['date']
    iam_token = response['Item']['value']
    date_now = str(datetime.datetime.now())

    if date_now.split()[0] == date_reg.split()[0] and len(iam_token) != 0 and \
            abs(int(date_now.split()[1].split(':')[0]) - int(date_reg.split()[1].split(':')[0])) <= 1:
                return iam_token






    oath_token = table.get_item(Key={'token_name': str('oath_token')})['Item']['value']
    iam_token = get_iam_token_from_oath(oath_token)

    response = table.update_item(
        Key={'token_name': 'iam_token'},
        UpdateExpression="set value=:v, date=:d",
        ExpressionAttributeValues={
            ':v': iam_token,
            ':d': date_now
        }
    )



    return iam_token