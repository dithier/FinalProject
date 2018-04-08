import requests
URL= "https://www.thebluealliance.com/api/v3/event/2018nhdur/teams"
auth_key = 'fge7icVbwIkRUkYKFb7Bj045jGELlWspOnCTxJnhkC9jqiLRjE0VBR4ACcez4vxo'

headers = {"X-TBA-Auth-Key": auth_key}
response = requests.get(URL, headers = headers)
data = response.json

# find the rookie year and team number of the first listed team
rookieyr = data[0]["team_number"]
teamnum = data[0]["team_number"]

""" 
Other URLs I will have to parse

Alliance team numbers and scores from:
https://www.thebluealliance.com/api/v3/event/2018nhdur/matches/simple

team numbers and sort orders from:
https://www.thebluealliance.com/api/v3/event/2018nhdur/rankings
"""