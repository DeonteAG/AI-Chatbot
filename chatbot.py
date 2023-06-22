
import random
import pandas
import json
import numpy
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests


# Simple chatbot Human Artificial Intelligence
# Multifunctional Chatbot
# Student ID: efyda1 
# Name: Deonte Allen Gooden


def main():
    
    loadData()
    botOnline = True
    botName = 'Messi'
    print(f"{botName}: Hi!, I am a multifunctional bot here to assist you :)")
    username = setUsername(botName)
    print(f"{botName}: Hi {username}, I'm {botName} but some call me the GOAT, lets's get started if you need help or want to exit let me know!.")

    
    while(botOnline):

     userInput = input(f"{username}: ")
     intents = intentMatcher(userInput.lower(), botName)
     
     if(intents != None):

          if(intents == 'terminate'):
               print(f"{botName}: Goodbye, {username} I hope you had a good expeirence :)") 
               botOnline = False 
          elif(intents == 'helpMessage'):
               print(f'''{botName}: I am a multifunctional bot, you can ask me the date, time or we can talk!.\nWe can play a football quiz or you can ask me some random questions? ps I love football.\nYou can change you name at any point''')

          elif(intents == 'football_quiz'):
               print(f"{botName}: You want to play the quiz Let go then!")
               footballQuiz(username, botName)

          elif(intents == 'play_game'):
               game(username,botName)

          elif(intents == 'weather'):
               print(f"{botName}: What city would you like to know the weather of?")
               
               print(f"{botName}: {get_weather(username,botName)}")

          elif(intents == 'changeName'):
               identity = nameChange(intents, botName, username)
               name = identity[0]
               if(identity[1] != None):
                    username = identity[1]
               print(f"{botName}: {name}")
          
          else:
               print(f"{botName}:{smallTalk(intents, botName, username)}")
     else:
          print(f"{botName}: {questionDatabase(userInput,botName)}")

def get_weather(username, botName):
     # Define the city for which to get weather data
     city = input(f"{username}: ")

     # Define the URL of the web page to scrape
     url = f"https://www.wunderground.com/weather/us/ny/{city.lower()}"

     # Make a GET request to the web page
     response = requests.get(url)

     # Parse the HTML content of the web page
     soup = BeautifulSoup(response.text, "html.parser")

     # Find the element containing the current temperature
     temp_element = soup.find("span", {"class": "wu-value wu-value-to"})

     # Extract the temperature from the element
     temp_str = temp_element.text.strip()
     temp_celsius = (float(temp_str) - 32) * 5/9

     # Print the current temperature in Celsius
     weather = print(f"{botName}: The current temperature in {city} is {temp_celsius:.1f} Celsius")

     return weather
          

def setUsername(botName):
   
   print(f"{botName}: What should I call you?")

   userInput = input(f"User: ")
   username = tokenise(userInput)

   if(userInput == ''): 
       username = 'User'
       print(f"{botName}: Your name has been set to User. Let me know at anytime if you want to change your name.")
   else:
     return username
     #print(f"{botName}: Hi {username}, I'm {botName} but some call me the GOAT, lets's get started if you need help or want to exit let me know!.")

   return username


#code retrieved from the lab work
def tokenise(userInput):

     """ method used for tokenising sentences and text retrieved from lab 0 work and the nltk """

     text = userInput
     text_tokens = word_tokenize(text)
     text_tokens_list = [ word . lower () for word in text_tokens if not word in
          stopwords . words () ]
     #print ( username )
     username = (" ").join (text_tokens_list)

     return username


def loadData():
     
     #converting my main dataset and Intents file in json files
     Dataset = pandas.read_csv('Datasets/Data.csv', encoding= 'unicode_escape')

     Intent_match = pandas.read_csv('Datasets/Intents.csv', encoding= 'unicode_escape')
     Dataset.drop(['QuestionID','Document'], axis = 1)

     Dataset_JSON = json.loads(Dataset.to_json(orient = 'records'))
     Intent_match_JSON = json.loads(Intent_match.to_json(orient='records'))

     with open("Datasets/new/Dataset.json", 'w') as New_Dataset:
          json.dump(Dataset_JSON, New_Dataset)
     with open("Datasets/new/Intents.json", 'w') as New_Intent:
          json.dump(Intent_match_JSON, New_Intent)


def intentMatcher(textInput, botName):
     """ this section matches up the intent of the user using cosine and maximum similarity with a threshold
     this code give the user a suggestion if the similarity is quite close """

     
     intentFile = 'Datasets/new/Intents.json'
     processedInput = inputPreprocessor(textInput, stopWordsF= False)
     words = filePreprocessor(file='Datasets/new/Intents.json', col = 'patterns', stopWordsF=False)
     

     cosineSimilarity, maximumSimilarity = similarities(words,processedInput)

     if(maximumSimilarity > 0.9):  
          index = numpy.where(cosineSimilarity == maximumSimilarity)[0][0]+2

          j = 0
          with open(intentFile, "r") as sentences_file:
               reader = json.load(sentences_file)
               for r in reader:
                    j += 1 
                    if j == index: 
                         return r['intents']
                         break
     
     elif  0.75 < maximumSimilarity < 0.9:
          index = numpy.where(cosineSimilarity == maximumSimilarity)[0][0]+2
          k = 0
          with open(intentFile, "r") as sentences_file:
               reader = json.load(sentences_file)
               for r in reader:
                    k += 1 
                    if k == index: 
                         suggestion = r['intents']
                         choice = input(f"{botName}: Were you asking about {suggestion} yes/no: ")
                         if choice.lower() == 'y' or choice.lower() == 'yes':
                              return r['intents']
                         elif choice.lower() == 'n' or choice.lower() == 'no':
                              print(f"{botName}: Sorry I do not know what you mean")
                              return
                         else:
                              print(f"{botName}: Invalid input")
                              
                              return
     
     else:
          return None


def lemmatise(text):
     
     """   lemmatisation method retrieved from lab0 """
     lemmatised_tokens = []
     lemmatiser = WordNetLemmatizer()

     for token in word_tokenize (text):
          lemmatised_tokens.append(lemmatiser.lemmatize(token))

     return lemmatised_tokens

def regexp(text):
     """ punctuation remover code retrieved from the nltk """

     tokenizer = RegexpTokenizer(r"\w+")  
     return[t for t in tokenizer.tokenize(text)]

def filePreprocessor(file, col, stopWordsF):

     """ section of the code processes the text ready for user input comparison """

     englishSW = stopwords.words('english')

     with open(file) as sFile:
        newSentences = []
        r = json.load(sFile)

        for rows in r:
            t = regexp((rows[col].lower()))

            fSentence = [] 
            if stopWordsF:
                fSentence = [w for w in t
                                        if w not in englishSW] 
            else:
                for d in t: 
                    fSentence.append(d)  

            fSentence = " ".join(fSentence).lower()
            newTokens = lemmatise(fSentence)
            fSentence = " ".join(newTokens).lower()
            newSentences.append(fSentence)    
            
        return newSentences
            
    

def inputPreprocessor(newInput, stopWordsF):

     """ section of the code processes the users input ready for text comparison """

     t = regexp(newInput.lower())
     englishSW = stopwords.words('english')
    
     if stopWordsF:
          fSentence = [w for w in t
                              if w not in englishSW] 
     else:
          fSentence = []
          for w in t: 
               fSentence.append(w)  
     
     fSentence = " ".join(fSentence).lower()
     newTokens = lemmatise(fSentence)    
     fSentence = " ".join(newTokens).lower()
     
     userInput = (fSentence, "")
     
     return userInput


def similarities(text, userInput):

     #Calculate the similarities with cosine similarities and max similiarities
     #retrieved and inspired from the lab activities and research
     tfidfVect = TfidfVectorizer ()
     X_train_counts = tfidfVect.fit_transform(text)
     X_train_input = tfidfVect.transform(userInput)
     cosineSimilarity = numpy.delete(cosine_similarity(X_train_input,X_train_counts),0)
     maximumSimilarity = cosineSimilarity.max()
     return cosineSimilarity, maximumSimilarity

 

def nameChange(intents, botName, username):
     if (intents == 'changeName'):
          newUsername = setUsername(botName)
          changeUsername = [f"Your name has now been set to {newUsername}", f"Okay I will now call you {newUsername} ",f"Hey, What's up {newUsername}"]
          response = random.choice(changeUsername)
          username = newUsername
          return response, username
     return username   


     
def smallTalk(intents, botName, username):
     
     """Responds to small talk depending on the intents and returns a response"""     
     t = time.time()
     
     if (intents == 'greetings'):
          greetings = ["Heyy, how may I assist you", "Hiya, Ask me a question :)", "Hello, how can i help you?"]
          response = random.choice(greetings) 
          return response

     elif (intents == 'thank_you'):
          thankYou = ["No Problem ;)","Your most welcome :D","It's okay", "Anytime!", "I am happy to help!", "My pleasure! XD"]
          response = random.choice(thankYou) 
          return response
          
     elif (intents == 'time'):
          timeReply = ["The time is ", "The time now is ","It is now ", "The current time is "]
          response = random.choice(timeReply) + time.strftime('%H:%M', time.localtime(t))
          return response
          
     elif (intents == 'date'):
          dateReply = ["The date is ", "The date today is ", "The current date is ", "Today it's "]
          response = random.choice(dateReply) + time.strftime('%A, %d %B %Y', time.localtime(t))
          return response

     """ In this section of the code I will be managing the user and the bots identity  """
     if (intents == 'askBotname'):
          askBotname = [f"Heyy, my name is {botName}", f"You can call me {botName}", f"Heyy, I'm {botName}"]
          response = random.choice(askBotname) 
          return response

     elif (intents == 'askUsername'):
          askUsername = [f"Don't be silly we go way back {username}",f"Your name is {username} :D",f"You told me to call you {username}"]
          response = random.choice(askUsername) 
          return response

     

def questionDatabase(textInput, botName):
     """ 
     This section of code uses my dataset and matches the users input with the questions in the questions database.
     If it is then confirmed the answer and the question with the cloest match will pop up
     This uses a mix of content from lab2 and lab3 alond with external research  
     """

     #code was inspired from the lab activities and similarity research
     intentFile = 'Datasets/new/Dataset.json'
     processedInput = inputPreprocessor(textInput, stopWordsF= False)
     words = filePreprocessor(file='Datasets/new/Dataset.json', col = 'Question', stopWordsF=False)
     

     cosineSimilarity, maximumSimilarity = similarities(words,processedInput)

     if(maximumSimilarity > 0.8):  
          index = numpy.where(cosineSimilarity == maximumSimilarity)[0][0]+2

          j = 0
          with open(intentFile, "r") as sentences_file:
               reader = json.load(sentences_file)
               for r in reader:
                    j += 1 
                    if j == index: 
                         return r['Answer']
                         
     
     elif  0.4 < maximumSimilarity < 0.8:
          index = numpy.where(cosineSimilarity == maximumSimilarity)[0][0]+2
          k = 0
          with open(intentFile, "r") as sentences_file:
               reader = json.load(sentences_file)
               for r in reader:
                    k += 1 
                    if k == index: 
                         suggestion = r['Document']
                         choice = input(f"{botName}: Is your Question {suggestion} related? yes/no: ")
                         if choice.lower() == 'yes' or choice.lower() == 'y':
                              return r['Answer']
                         elif choice.lower() == 'no' or choice.lower() == 'n':
                              print(f"{botName}: Sorry I am not sure what you mean :(")
                              return
                         else:
                              print(f"{botName}: Invalid input")
                              return
     
     else:
          confused = "I do not know that question please can you rephrase yourself or ask me for help!"
          return confused
     
def footballQuiz(username, botName):
     """ This is a quick quiz that uses some of the question from the questions and answers database """

     quizScore = 0 
     quizQuestions = 4

     quizActive = True

     while(quizActive == True):
          """ Q1 """
          print(f"{botName}: Jurgen Klopp has managed two clubs in Germany, Borussia Dortmund and - can you name the other? ")
          answer = input(f"{username}: ")

          if answer.lower() == "mainz":
               print(f"{botName}: That's Correct!")
               quizScore += 1
          else:
               print(f"{botName}: Incorrect!, ask me after the quiz to find out :)")

          """ Q2 """
          print(f"{botName}: In video game FIFA 20, team Piemonte Calcio represents which real-life club? ")
          answer = input(f"{username}: ")

          if answer.lower() == "juventus":
               print(f"{botName}: That's Correct!")
               quizScore += 1
          else:
               print(f"{botName}: Incorrect!, ask me after the quiz to find out :)")

          
          """ Q3 """
          print(f"{botName}: Which one of the following three teams has not won the European Championship: Denmark, Belgium or Greece? ")
          answer = input(f"{username}: ")

          if answer.lower() == "belgium":
               print(f"{botName}: That's Correct!")
               quizScore += 1
          else:
               print(f"{botName}: Incorrect!, ask me after the quiz to find out :)")

          
          """ Q4 """
          print(f"{botName}: With 260 goals, who is the Premier League's all-time top scorer? ")
          answer = input(f"{username}: ")

          if answer.lower() == "alan shearer" or "alanshearer" or "as":
               print(f"{botName}: That's Correct!")
               quizScore += 1
          else:
               print(f"{botName}: Incorrect!, ask me after the quiz to find out :)")

          print(f"{botName}: Quiz complete you scored: {quizScore}/{quizQuestions} ")
          quizActive = False
     return None


def game(username, botName):
     """ Here is I will allow the user to play a quick number guessing game """

     print(f"{botName}: Welcome to the number guessing game!")

     correctNumber = random.randint(1,10)

     print(f"{botName}: I am thinking of a number between 1 and 10. Can you guess what it is?")

     # Allow the player to make up to 3 guesses
     for guess_count in range(3):
          guess = input(f"{username}: ")
          if guess == correctNumber:
               print(f"{botName}: Congratulations, {username}! You guessed the correct number!")
               break

          else:
               print(f"{botName}: Sorry, {username}. That is not the correct number.")
               if guess_count == 2:
                    print(f"Sorry, {username}. You have run out of guesses. The correct number was {correctNumber}.")

  
main()