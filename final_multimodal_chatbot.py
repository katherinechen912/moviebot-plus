"""
MovieBot - Final Project Multimodal Chatbot
Features from original MovieBot:
- Movie recommendations from IMDb Top 100 list
- Movie info via OMDb API
- Famous quotes & aphorisms
- Sentiment analysis
- Image processing capabilities
"""

import random
import re
import nltk
import os
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
from nltk.corpus import movie_reviews, wordnet
from nltk.tokenize import word_tokenize
import nltk
nltk.data.path.append("nltk_data")


# First-time setup: Uncomment and run once if needed
# nltk.download('movie_reviews')
# nltk.download('punkt')
# nltk.download('wordnet')

# ------------------ API Setup ------------------
OMDB_API_KEY = '5a48156b'  # API key for OMDb API

# ------------------ Create directories for images ------------------
PRESET_DIR = "preset_images"
USER_DIR = "user_images"
PROCESSED_DIR = "processed_images"

# Create directories if they don't exist
for directory in [PRESET_DIR, USER_DIR, PROCESSED_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ------------------ Greeting and Farewell Setup ------------------
GREETING_KEYWORDS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["'sup", "hey", "*tips hat*", "Hey there, movie fan!", "Welcome to MovieBot!"]

FAREWELL_KEYWORDS = ("bye", "goodbye", "see you", "quit", "exit")
FAREWELL_RESPONSES = ["See you later!", "Have a great day!", "Until next time!", "Come back for more movie talk!", "Goodbye!"]

# ------------------ IMDb Top Movies (Full IMDb Top 100) ------------------
imdb_top_movies = [ # Movies #1–10
    {"title": "The Shawshank Redemption", "year": 1994, "rating": 9.3, "director": "Frank Darabont", "genre": ["Drama"],
     "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."},
    {"title": "The Godfather", "year": 1972, "rating": 9.2, "director": "Francis Ford Coppola", "genre": ["Crime", "Drama"],
     "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
    {"title": "The Dark Knight", "year": 2008, "rating": 9.0, "director": "Christopher Nolan", "genre": ["Action", "Crime", "Drama"],
     "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
    {"title": "12 Angry Men", "year": 1957, "rating": 9.0, "director": "Sidney Lumet", "genre": ["Drama", "Crime"],
     "plot": "A jury holdout attempts to prevent a miscarriage of justice by forcing his colleagues to reconsider the evidence."},
    {"title": "Schindler's List", "year": 1993, "rating": 9.0, "director": "Steven Spielberg", "genre": ["Biography", "Drama", "History"],
     "plot": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution by the Nazis."},
    {"title": "The Godfather Part II", "year": 1974, "rating": 9.0, "director": "Francis Ford Coppola", "genre": ["Crime", "Drama"],
     "plot": "The early life and career of Vito Corleone in 1920s New York City is portrayed, while his son, Michael, expands and tightens his grip on the family crime syndicate."},
    {"title": "The Lord of the Rings: The Return of the King", "year": 2003, "rating": 9.0, "director": "Peter Jackson", "genre": ["Adventure", "Drama", "Fantasy"],
     "plot": "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring."},
    {"title": "Pulp Fiction", "year": 1994, "rating": 8.9, "director": "Quentin Tarantino", "genre": ["Crime", "Drama"],
     "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
    {"title": "The Lord of the Rings: The Fellowship of the Ring", "year": 2001, "rating": 8.9, "director": "Peter Jackson", "genre": ["Adventure", "Drama", "Fantasy"],
     "plot": "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron."},
    {"title": "Inception", "year": 2010, "rating": 8.8, "director": "Christopher Nolan", "genre": ["Action", "Adventure", "Sci-Fi"],
     "plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."}
]

more_movies_11_30 = [
    {"title": "Fight Club", "year": 1999, "rating": 8.8, "director": "David Fincher", "genre": ["Drama"],
     "plot": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more."},

    {"title": "Forrest Gump", "year": 1994, "rating": 8.8, "director": "Robert Zemeckis", "genre": ["Drama", "Romance"],
     "plot": "The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate and other historical events unfold through the perspective of an Alabama man with an IQ of 75."},

    {"title": "The Good, the Bad and the Ugly", "year": 1966, "rating": 8.8, "director": "Sergio Leone", "genre": ["Western"],
     "plot": "A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a fortune in gold buried in a remote cemetery."},

    {"title": "The Lord of the Rings: The Two Towers", "year": 2002, "rating": 8.8, "director": "Peter Jackson", "genre": ["Adventure", "Drama", "Fantasy"],
     "plot": "While Frodo and Sam edge closer to Mordor with the help of the shifty Gollum, the divided fellowship makes a stand against Sauron's new ally, Saruman."},

    {"title": "12th Fail", "year": 2023, "rating": 8.8, "director": "Vidhu Vinod Chopra", "genre": ["Biography", "Drama"],
     "plot": "Based on the real-life story of IPS officer Manoj Kumar Sharma and his journey of resilience and determination."},

    {"title": "Hababam Sinifi: Sinifta Kaldi", "year": 1975, "rating": 8.8, "director": "Ertem Eğilmez", "genre": ["Comedy"],
     "plot": "The misadventures of a group of students who refuse to graduate from their school."},

    {"title": "Interstellar", "year": 2014, "rating": 8.7, "director": "Christopher Nolan", "genre": ["Adventure", "Drama", "Sci-Fi"],
     "plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},

    {"title": "The Matrix", "year": 1999, "rating": 8.7, "director": "Lana and Lilly Wachowski", "genre": ["Action", "Sci-Fi"],
     "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."},

    {"title": "Goodfellas", "year": 1990, "rating": 8.7, "director": "Martin Scorsese", "genre": ["Crime", "Drama"],
     "plot": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen and his partners Jimmy and Tommy."},

    {"title": "One Flew Over the Cuckoo's Nest", "year": 1975, "rating": 8.7, "director": "Milos Forman", "genre": ["Drama"],
     "plot": "A criminal pleads insanity and is admitted to a mental institution, where he rebels against the oppressive nurse and rallies the scared patients."},

    {"title": "Star Wars: Episode V - The Empire Strikes Back", "year": 1980, "rating": 8.7, "director": "Irvin Kershner", "genre": ["Action", "Adventure", "Fantasy"],
     "plot": "After the Rebels are overpowered by the Empire on the ice planet Hoth, Luke begins Jedi training with Yoda while his friends are pursued by Darth Vader."},

    {"title": "Se7en", "year": 1995, "rating": 8.6, "director": "David Fincher", "genre": ["Crime", "Drama", "Mystery"],
     "plot": "Two detectives hunt a serial killer who uses the seven deadly sins as his motives."},

    {"title": "The Silence of the Lambs", "year": 1991, "rating": 8.6, "director": "Jonathan Demme", "genre": ["Crime", "Drama", "Thriller"],
     "plot": "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to catch another serial killer."},

    {"title": "Spirited Away", "year": 2001, "rating": 8.6, "director": "Hayao Miyazaki", "genre": ["Animation", "Adventure", "Family"],
     "plot": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits."},

    {"title": "The Green Mile", "year": 1999, "rating": 8.6, "director": "Frank Darabont", "genre": ["Crime", "Drama", "Fantasy"],
     "plot": "The lives of guards on Death Row are affected by one of their charges: a black man accused of murder who has a mysterious gift."},

    {"title": "Terminator 2: Judgment Day", "year": 1991, "rating": 8.6, "director": "James Cameron", "genre": ["Action", "Sci-Fi"],
     "plot": "A cyborg, identical to the one who failed to kill Sarah Connor, must now protect her teenage son from a more advanced and powerful Terminator."},

    {"title": "Star Wars", "year": 1977, "rating": 8.6, "director": "George Lucas", "genre": ["Action", "Adventure", "Fantasy"],
     "plot": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire’s battle station."},

    {"title": "Saving Private Ryan", "year": 1998, "rating": 8.6, "director": "Steven Spielberg", "genre": ["Drama", "War"],
     "plot": "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action."},

    {"title": "City of God", "year": 2002, "rating": 8.6, "director": "Fernando Meirelles, Kátia Lund", "genre": ["Crime", "Drama"],
     "plot": "In the slums of Rio, two kids' paths diverge as one struggles to become a photographer and the other a kingpin."},

    {"title": "Life Is Beautiful", "year": 1997, "rating": 8.6, "director": "Roberto Benigni", "genre": ["Comedy", "Drama", "Romance"],
     "plot": "When an open-minded Jewish librarian and his son become victims of the Holocaust, he uses humor and imagination to protect his son."}
]

more_movies_31_50 = [
    {"title": "Seven Samurai", "year": 1954, "rating": 8.6, "director": "Akira Kurosawa",
     "genre": ["Action", "Drama"],
     "plot": "Farmers from a village exploited by bandits hire a veteran samurai for protection, and he gathers six other samurai to join him."},

    {"title": "It's a Wonderful Life", "year": 1946, "rating": 8.6, "director": "Frank Capra",
     "genre": ["Drama", "Family", "Fantasy"],
     "plot": "An angel is sent from Heaven to help a desperately frustrated businessman by showing him what life would have been like if he had never existed."},

    {"title": "Harakiri", "year": 1962, "rating": 8.6, "director": "Masaki Kobayashi",
     "genre": ["Action", "Drama", "Mystery"],
     "plot": "When a ronin requesting seppuku at a feudal lord's palace is told of the brutal suicide of another ronin who previously visited, he reveals how their pasts are intertwined - and in doing so challenges the clan's integrity."},

    {"title": "Dune: Part Two", "year": 2024, "rating": 8.5, "director": "Denis Villeneuve",
     "genre": ["Action", "Adventure", "Drama"],
     "plot": "Paul Atreides unites with the Fremen while on a warpath of revenge against the conspirators who destroyed his family. Facing a choice between the love of his life and the fate of the universe, he endeavors to prevent a terrible future."},

    {"title": "Gladiator", "year": 2000, "rating": 8.5, "director": "Ridley Scott",
     "genre": ["Action", "Adventure", "Drama"],
     "plot": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."},

    {"title": "Parasite", "year": 2019, "rating": 8.5, "director": "Bong Joon Ho",
     "genre": ["Drama", "Thriller"],
     "plot": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."},

    {"title": "The Prestige", "year": 2006, "rating": 8.5, "director": "Christopher Nolan",
     "genre": ["Drama", "Mystery", "Sci-Fi"],
     "plot": "Rival 19th-century magicians engage in a bitter battle for trade secrets."},

    {"title": "Django Unchained", "year": 2012, "rating": 8.5, "director": "Quentin Tarantino",
     "genre": ["Drama", "Western"],
     "plot": "With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal plantation owner in Mississippi."},

    {"title": "The Lion King", "year": 1994, "rating": 8.5, "director": "Roger Allers, Rob Minkoff",
     "genre": ["Animation", "Adventure", "Drama"],
     "plot": "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself."},

    {"title": "The Departed", "year": 2006, "rating": 8.5, "director": "Martin Scorsese",
     "genre": ["Crime", "Drama", "Thriller"],
     "plot": "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston."},

    {"title": "Spider-Man: Across the Spider-Verse", "year": 2023, "rating": 8.5,
     "director": "Joaquim Dos Santos, Kemp Powers, Justin K. Thompson",
     "genre": ["Animation", "Action", "Adventure"],
     "plot": "Traveling across the multiverse, Miles Morales meets a new team of Spider-People, made up of heroes from different dimensions. But when the heroes clash over how to deal with a new threat, Miles finds himself at a crossroads."},

    {"title": "Whiplash", "year": 2014, "rating": 8.5, "director": "Damien Chazelle",
     "genre": ["Drama", "Music"],
     "plot": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential."},

    {"title": "Léon: The Professional", "year": 1994, "rating": 8.5, "director": "Luc Besson",
     "genre": ["Action", "Crime", "Drama"],
     "plot": "An Italian hitman protects a young girl whose family was killed by corrupt DEA agents."},

    {"title": "Back to the Future", "year": 1985, "rating": 8.5, "director": "Robert Zemeckis",
     "genre": ["Adventure", "Comedy", "Sci-Fi"],
     "plot": "Marty McFly, a 17-year-old high school student, is accidentally sent 30 years into the past in a time-traveling DeLorean invented by his close friend, the maverick scientist Doc Brown."},

    {"title": "Alien", "year": 1979, "rating": 8.5, "director": "Ridley Scott",
     "genre": ["Horror", "Sci-Fi"],
     "plot": "After investigating a mysterious transmission of unknown origin, the crew of a commercial spacecraft encounters a deadly lifeform."},

    {"title": "American History X", "year": 1998, "rating": 8.5, "director": "Tony Kaye",
     "genre": ["Drama"],
     "plot": "A former neo-nazi tries to prevent his younger brother from going down the same path of hate he once did."},

    {"title": "The Usual Suspects", "year": 1995, "rating": 8.5, "director": "Bryan Singer",
     "genre": ["Crime", "Mystery", "Thriller"],
     "plot": "The sole survivor of a pier shoot-out tells the story of how a notorious criminal influenced the events that began with five criminals meeting in a seemingly random police lineup."},

    {"title": "Grave of the Fireflies", "year": 1988, "rating": 8.5, "director": "Isao Takahata",
     "genre": ["Animation", "Drama", "War"],
     "plot": "A young boy and his little sister struggle to survive in Japan during World War II."},

    {"title": "The Pianist", "year": 2002, "rating": 8.5, "director": "Roman Polanski",
     "genre": ["Biography", "Drama", "Music"],
     "plot": "A Polish Jewish musician struggles to survive the destruction of the Warsaw ghetto of World War II."},

    {"title": "Once Upon a Time in the West", "year": 1968, "rating": 8.5, "director": "Sergio Leone",
     "genre": ["Western"],
     "plot": "A mysterious stranger with a harmonica joins forces with a notorious desperado to protect a beautiful widow from a ruthless assassin working for the railroad."}
]

more_movies_51_70 = [
    {"title": "Intouchables", "year": 2011, "rating": 8.5, "director": "Olivier Nakache, Éric Toledano",
     "genre": ["Biography", "Comedy", "Drama"],
     "plot": "After he becomes a quadriplegic from a paragliding accident, an aristocrat hires a young man from the projects to be his caregiver."},
    {"title": "Psycho", "year": 1960, "rating": 8.5, "director": "Alfred Hitchcock",
     "genre": ["Horror", "Mystery", "Thriller"],
     "plot": "A secretary embezzles money and checks into a remote motel run by a young man under the domination of his mother."},
    {"title": "Casablanca", "year": 1942, "rating": 8.5, "director": "Michael Curtiz",
     "genre": ["Drama", "Romance", "War"],
     "plot": "A cynical expatriate American cafe owner struggles to decide whether or not to help his former lover and her fugitive husband escape the Nazis in French Morocco."},
    {"title": "Rear Window", "year": 1954, "rating": 8.5, "director": "Alfred Hitchcock",
     "genre": ["Mystery", "Thriller"],
     "plot": "A wheelchair-bound photographer spies on his neighbors from his apartment window and becomes convinced one of them has committed murder."},
    {"title": "Cinema Paradiso", "year": 1988, "rating": 8.5, "director": "Giuseppe Tornatore",
     "genre": ["Drama", "Romance"],
     "plot": "A filmmaker recalls his childhood when falling in love with the pictures at the cinema of his home village and forms a deep friendship with the cinema's projectionist."},
    {"title": "Modern Times", "year": 1936, "rating": 8.5, "director": "Charles Chaplin",
     "genre": ["Comedy", "Drama", "Family"],
     "plot": "The Tramp struggles to live in modern industrial society with the help of a young homeless woman."},
    {"title": "City Lights", "year": 1931, "rating": 8.5, "director": "Charles Chaplin",
     "genre": ["Comedy", "Drama", "Romance"],
     "plot": "With the aid of a wealthy erratic tippler, a dewy-eyed tramp who has fallen in love with a sightless flower girl accumulates money to help her medically."},
    {"title": "Inglourious Basterds", "year": 2009, "rating": 8.4, "director": "Quentin Tarantino",
     "genre": ["Adventure", "Drama", "War"],
     "plot": "In Nazi-occupied France, a group of Jewish U.S. soldiers plan to assassinate Nazi leaders while a theater owner seeks revenge for the death of her family."},
    {"title": "Avengers: Endgame", "year": 2019, "rating": 8.4, "director": "Anthony Russo, Joe Russo",
     "genre": ["Action", "Adventure", "Drama"],
     "plot": "After the devastating events of Avengers: Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more to undo Thanos's actions."},
    {"title": "Avengers: Infinity War", "year": 2018, "rating": 8.4, "director": "Anthony Russo, Joe Russo",
     "genre": ["Action", "Adventure", "Sci-Fi"],
     "plot": "The Avengers and their allies must be willing to sacrifice all in an attempt to defeat the powerful Thanos before his blitz of devastation puts an end to the universe."},
    {"title": "Apocalypse Now", "year": 1979, "rating": 8.4, "director": "Francis Ford Coppola",
     "genre": ["Drama", "War"],
     "plot": "A U.S. Army officer is sent on a mission into Cambodia to assassinate a renegade colonel who has set himself up as a god among a local tribe."},
    {"title": "The Dark Knight Rises", "year": 2012, "rating": 8.4, "director": "Christopher Nolan",
     "genre": ["Action", "Drama"],
     "plot": "Eight years after the Joker's reign, Batman encounters a new terrorist leader, Bane, who overwhelms Gotham's finest and forces the Dark Knight to resurface."},
    {"title": "Memento", "year": 2000, "rating": 8.4, "director": "Christopher Nolan",
     "genre": ["Mystery", "Thriller"],
     "plot": "A man with short-term memory loss attempts to track down his wife's murderer using notes and tattoos to remember facts about himself."},
    {"title": "The Shining", "year": 1980, "rating": 8.4, "director": "Stanley Kubrick",
     "genre": ["Drama", "Horror"],
     "plot": "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence, while his psychic son sees horrific forebodings."},
    {"title": "Amadeus", "year": 1984, "rating": 8.4, "director": "Milos Forman",
     "genre": ["Biography", "Drama", "Music"],
     "plot": "The life, success, and troubles of Wolfgang Amadeus Mozart, as told by Antonio Salieri, the contemporaneous composer who was deeply jealous of Mozart's talent."},
    {"title": "Aliens", "year": 1986, "rating": 8.4, "director": "James Cameron",
     "genre": ["Action", "Adventure", "Sci-Fi"],
     "plot": "Ellen Ripley is rescued after drifting in space for 57 years and returns to the planet where her crew encountered the alien creature, this time with space marines."},
    {"title": "Raiders of the Lost Ark", "year": 1981, "rating": 8.4, "director": "Steven Spielberg",
     "genre": ["Action", "Adventure"],
     "plot": "Archaeologist Indiana Jones is hired by the U.S. government to find the Ark of the Covenant before the Nazis can obtain its awesome powers."},
    {"title": "Spider-Man: Into the Spider-Verse", "year": 2018, "rating": 8.4, "director": "Bob Persichetti, Peter Ramsey, Rodney Rothman",
     "genre": ["Animation", "Action", "Adventure"],
     "plot": "Teen Miles Morales becomes Spider-Man of his reality, crossing paths with five counterparts from other dimensions to stop a threat for all realities."},
    {"title": "Your Name", "year": 2016, "rating": 8.4, "director": "Makoto Shinkai",
     "genre": ["Animation", "Drama", "Fantasy"],
     "plot": "Two teenagers share a profound, magical connection upon discovering they are swapping bodies. Things become more complicated when they decide to meet in person."},
    {"title": "Coco", "year": 2017, "rating": 8.4, "director": "Lee Unkrich, Adrian Molina",
     "genre": ["Animation", "Adventure", "Family"],
     "plot": "Aspiring musician Miguel, confronted with his family's ancestral ban on music, enters the Land of the Dead to find his great-great-grandfather, a legendary singer."}
]

more_movies_71_90 = [
    {"title": "WALL·E", "year": 2008, "rating": 8.4, "director": "Andrew Stanton",
     "genre": ["Animation", "Adventure", "Family"],
     "plot": "In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind."},
    {"title": "The Lives of Others", "year": 2006, "rating": 8.4, "director": "Florian Henckel von Donnersmarck",
     "genre": ["Drama", "Mystery", "Thriller"],
     "plot": "In 1984 East Berlin, an agent of the secret police becomes absorbed in the lives of a couple he's surveilling and is drawn into their world."},
    {"title": "3 Idiots", "year": 2009, "rating": 8.4, "director": "Rajkumar Hirani",
     "genre": ["Comedy", "Drama"],
     "plot": "Two friends are searching for their long-lost companion. They revisit their college days and recall the memories of their friend who inspired them to think differently."},
    {"title": "Capernaum", "year": 2018, "rating": 8.4, "director": "Nadine Labaki",
     "genre": ["Drama"],
     "plot": "While serving a five-year sentence for a violent crime, a 12-year-old boy sues his parents for neglect."},
    {"title": "Das Boot", "year": 1981, "rating": 8.4, "director": "Wolfgang Petersen",
     "genre": ["Drama", "War"],
     "plot": "The claustrophobic world of a WWII German U-boat; boredom, filth, and sheer terror."},
    {"title": "Sunset Blvd.", "year": 1950, "rating": 8.4, "director": "Billy Wilder",
     "genre": ["Drama", "Film-Noir"],
     "plot": "A screenwriter develops a dangerous relationship with a faded film star determined to make a triumphant return."},
    {"title": "Witness for the Prosecution", "year": 1957, "rating": 8.4, "director": "Billy Wilder",
     "genre": ["Crime", "Drama", "Mystery"],
     "plot": "A veteran British barrister must defend his client in a murder trial that has surprise after surprise."},
    {"title": "Paths of Glory", "year": 1957, "rating": 8.4, "director": "Stanley Kubrick",
     "genre": ["Drama", "War"],
     "plot": "After refusing to attack an enemy position, a general accuses his soldiers of cowardice and their commanding officer must defend them."},
    {"title": "The Great Dictator", "year": 1940, "rating": 8.4, "director": "Charles Chaplin",
     "genre": ["Comedy", "Drama", "War"],
     "plot": "Dictator Adenoid Hynkel tries to expand his empire while a poor Jewish barber tries to avoid persecution."},
    {"title": "High and Low", "year": 1963, "rating": 8.4, "director": "Akira Kurosawa",
     "genre": ["Crime", "Drama", "Mystery"],
     "plot": "An executive of a shoe company becomes a victim of extortion when his chauffeur's son is kidnapped by mistake."},
    {"title": "Princess Mononoke", "year": 1997, "rating": 8.3, "director": "Hayao Miyazaki",
     "genre": ["Animation", "Adventure", "Fantasy"],
     "plot": "On a journey to find the cure for a curse, a prince finds himself in the middle of a war between a forest and humans who consume its resources."},
    {"title": "Good Will Hunting", "year": 1997, "rating": 8.3, "director": "Gus Van Sant",
     "genre": ["Drama", "Romance"],
     "plot": "Will Hunting, a janitor at M.I.T., has a gift for mathematics, but needs help from a psychologist to find direction in his life."},
    {"title": "American Beauty", "year": 1999, "rating": 8.3, "director": "Sam Mendes",
     "genre": ["Drama"],
     "plot": "A sexually frustrated suburban father has a mid-life crisis after becoming infatuated with his daughter's best friend."},
    {"title": "Eternal Sunshine of the Spotless Mind", "year": 2004, "rating": 8.3, "director": "Michel Gondry",
     "genre": ["Drama", "Romance", "Sci-Fi"],
     "plot": "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories."},
    {"title": "Requiem for a Dream", "year": 2000, "rating": 8.3, "director": "Darren Aronofsky",
     "genre": ["Drama"],
     "plot": "The drug-induced utopias of four Coney Island people are shattered when their addictions run deep."},
    {"title": "Oldboy", "year": 2003, "rating": 8.3, "director": "Park Chan-wook",
     "genre": ["Action", "Drama", "Mystery"],
     "plot": "After being kidnapped and imprisoned for 15 years, Oh Dae-Su is released, only to find that he must find his captor in five days."},
    {"title": "Braveheart", "year": 1995, "rating": 8.3, "director": "Mel Gibson",
     "genre": ["Biography", "Drama", "History"],
     "plot": "Scottish warrior William Wallace leads his countrymen in a rebellion to free his homeland from the tyranny of King Edward I of England."},
    {"title": "2001: A Space Odyssey", "year": 1968, "rating": 8.3, "director": "Stanley Kubrick",
     "genre": ["Adventure", "Sci-Fi"],
     "plot": "After discovering a mysterious artifact buried beneath the lunar surface, mankind sets off on a quest to find its origins with help from intelligent computer HAL 9000."},
    {"title": "Joker", "year": 2019, "rating": 8.3, "director": "Todd Phillips",
     "genre": ["Crime", "Drama", "Thriller"],
     "plot": "In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society. He begins a slow descent into madness."},
    {"title": "Come and See", "year": 1985, "rating": 8.3, "director": "Elem Klimov",
     "genre": ["Drama", "War"],
     "plot": "After finding an old rifle, a young boy joins the Soviet resistance movement against ruthless German forces and experiences the horrors of WWII."}
]

more_movies_91_100 = [
    {"title": "Reservoir Dogs", "year": 1992, "rating": 8.3, "director": "Quentin Tarantino",
     "genre": ["Crime", "Drama", "Thriller"],
     "plot": "After a simple jewelry heist goes terribly wrong, the surviving criminals begin to suspect that one of them is a police informant."},
    {"title": "Once Upon a Time in America", "year": 1984, "rating": 8.3, "director": "Sergio Leone",
     "genre": ["Crime", "Drama"],
     "plot": "A former Prohibition-era Jewish gangster returns to Brooklyn over 30 years later, where he once again must confront the ghosts and regrets of his old life."},
    {"title": "The Hunt", "year": 2012, "rating": 8.3, "director": "Thomas Vinterberg",
     "genre": ["Drama"],
     "plot": "A teacher lives a lonely life, all the while struggling over his son's custody. His life slowly gets better as he finds love and receives good news from his son—until a lie turns his life upside down."},
    {"title": "Toy Story", "year": 1995, "rating": 8.3, "director": "John Lasseter",
     "genre": ["Animation", "Adventure", "Comedy"],
     "plot": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room."},
    {"title": "Star Wars: Episode VI - Return of the Jedi", "year": 1983, "rating": 8.3, "director": "Richard Marquand",
     "genre": ["Action", "Adventure", "Fantasy"],
     "plot": "After rescuing Han Solo from Jabba the Hutt, the Rebels attempt to destroy the second Death Star, while Luke struggles to bring Vader back from the dark side."},
    {"title": "Singin' in the Rain", "year": 1952, "rating": 8.3, "director": "Gene Kelly, Stanley Donen",
     "genre": ["Comedy", "Musical", "Romance"],
     "plot": "A silent film production company and cast make a difficult transition to sound."},
    {"title": "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb", "year": 1964, "rating": 8.3, "director": "Stanley Kubrick",
     "genre": ["Comedy", "War"],
     "plot": "An insane general triggers a path to nuclear holocaust that a war room full of politicians and generals frantically try to stop."},
    {"title": "Toy Story 3", "year": 2010, "rating": 8.3, "director": "Lee Unkrich",
     "genre": ["Animation", "Adventure", "Comedy"],
     "plot": "The toys are mistakenly delivered to a day-care center instead of the attic right before Andy leaves for college, and it's up to Woody to convince the others they weren't abandoned and to return home."},
    {"title": "The Apartment", "year": 1960, "rating": 8.3, "director": "Billy Wilder",
     "genre": ["Comedy", "Drama", "Romance"],
     "plot": "A man tries to rise in his company by letting its executives use his apartment for trysts, but complications and a romance of his own ensue."},
    {"title": "Ikiru", "year": 1952, "rating": 8.3, "director": "Akira Kurosawa",
     "genre": ["Drama"],
     "plot": "A bureaucrat tries to find a meaning in his life after he discovers he has terminal cancer."}
]

# Combine the lists
imdb_top_movies.extend(more_movies_11_30)
imdb_top_movies.extend(more_movies_31_50)
imdb_top_movies.extend(more_movies_51_70)
imdb_top_movies.extend(more_movies_71_90)
imdb_top_movies.extend(more_movies_91_100)

# ------------------ Top 100 Movie Quotes ------------------
top_100_movie_quotes = [
    "A census taker once tried to test me. I ate his liver with some fava beans and a nice Chianti. – The Silence of the Lambs (1991)",
    "Bond. James Bond. – Dr. No (1962)",
    "There's no place like home. – The Wizard of Oz (1939)",
    "I am big! It's the pictures that got small. – Sunset Blvd. (1950)",
    "Show me the money! – Jerry Maguire (1996)",
    "Why don't you come up sometime and see me? – She Done Him Wrong (1933)",
    "I'm walking here! I'm walking here! – Midnight Cowboy (1969)",
    "Play it, Sam. Play 'As Time Goes By.' – Casablanca (1942)",
    "You can't handle the truth! – A Few Good Men (1992)",
    "I want to be alone. – Grand Hotel (1932)",
    "After all, tomorrow is another day! – Gone with the Wind (1939)",
    "Round up the usual suspects. – Casablanca (1942)",
    "I'll have what she's having. – When Harry Met Sally... (1989)",
    "You know how to whistle, don't you, Steve? You just put your lips together and blow. – To Have and Have Not (1944)",
    "You're gonna need a bigger boat. – Jaws (1975)",
    "Badges? We don't need no stinkin' badges! – The Treasure of the Sierra Madre (1948)",
    "I'll be back. – The Terminator (1984)",
    "Today, I consider myself the luckiest man on the face of the Earth. – The Pride of the Yankees (1942)",
    "If you build it, he will come. – Field of Dreams (1989)",
    "Mama always said life was like a box of chocolates. You never know what you're gonna get. – Forrest Gump (1994)",
    "We rob banks. – Bonnie and Clyde (1967)",
    "Plastics. – The Graduate (1967)",
    "We'll always have Paris. – Casablanca (1942)",
    "I see dead people. – The Sixth Sense (1999)",
    "Stella! Hey, Stella! – A Streetcar Named Desire (1951)",
    "Oh, Jerry, don't let's ask for the moon. We have the stars. – Now, Voyager (1942)",
    "Shane. Shane. Come back! – Shane (1953)",
    "Well, nobody's perfect. – Some Like It Hot (1959)",
    "It's alive! It's alive! – Frankenstein (1931)",
    "Houston, we have a problem. – Apollo 13 (1995)",
    "You've got to ask yourself one question: 'Do I feel lucky?' Well, do ya, punk? – Dirty Harry (1971)",
    "You had me at ‘hello’. – Jerry Maguire (1996)",
    "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't know. – Animal Crackers (1930)",
    "There's no crying in baseball! – A League of Their Own (1992)",
    "La-dee-da, la-dee-da. – Annie Hall (1977)",
    "A boy's best friend is his mother. – Psycho (1960)",
    "Greed, for lack of a better word, is good. – Wall Street (1987)",
    "Keep your friends close, but your enemies closer. – The Godfather Part II (1974)",
    "As God is my witness, I'll never be hungry again. – Gone with the Wind (1939)",
    "Well, here's another nice mess you've gotten me into! – Sons of the Desert (1933)",
    "Say 'hello' to my little friend! – Scarface (1983)",
    "What a dump. – Beyond the Forest (1949)",
    "Mrs. Robinson, you're trying to seduce me. Aren't you? – The Graduate (1967)",
    "Gentlemen, you can't fight in here! This is the War Room! – Dr. Strangelove (1964)",
    "Elementary, my dear Watson. – The Adventures of Sherlock Holmes (1939)",
    "Get your stinking paws off me, you damned dirty ape! – Planet of the Apes (1968)",
    "Of all the gin joints in all the towns in all the world, she walks into mine. – Casablanca (1942)",
    "Here's Johnny! – The Shining (1980)",
    "They're here! – Poltergeist (1982)",
    "Is it safe? – Marathon Man (1976)",
    "Wait a minute, wait a minute. You ain't heard nothin' yet! – The Jazz Singer (1927)",
    "No wire hangers, ever! – Mommie Dearest (1981)",
    "Mother of mercy, is this the end of Rico? – Little Caesar (1930)",
    "Forget it, Jake, it's Chinatown. – Chinatown (1974)",
    "I have always depended on the kindness of strangers. – A Streetcar Named Desire (1951)",
    "Hasta la vista, baby. – Terminator 2: Judgment Day (1991)",
    "Soylent Green is people! – Soylent Green (1973)",
    "Open the pod bay doors, HAL. – 2001: A Space Odyssey (1968)",
    "Striker: Surely you can't be serious. Rumack: I am serious... and don't call me Shirley. – Airplane! (1980)",
    "Yo, Adrian! – Rocky (1976)",
    "Hello, gorgeous. – Funny Girl (1968)",
    "Toga! Toga! – National Lampoon's Animal House (1978)",
    "Listen to them. Children of the night. What music they make. – Dracula (1931)",
    "Oh, no, it wasn't the airplanes. It was Beauty killed the Beast. – King Kong (1933)",
    "My precious. – The Lord of the Rings: The Two Towers (2002)",
    "Attica! Attica! – Dog Day Afternoon (1975)",
    "Sawyer, you're going out a youngster, but you've got to come back a star! – 42nd Street (1933)",
    "Listen to me, mister. You're my knight in shining armor. Don't you forget it. – On Golden Pond (1981)",
    "Tell 'em to go out there with all they got and win just one for the Gipper. – Knute Rockne All American (1940)",
    "A martini. Shaken, not stirred. – Goldfinger (1964)",
    "Who's on first. – The Naughty Nineties (1945)",
    "Cinderella story. Outta nowhere. A former greenskeeper, now, about to become the Masters champion. – Caddyshack (1980)",
    "Life is a banquet, and most poor suckers are starving to death! – Auntie Mame (1958)",
    "Snap out of it! – Moonstruck (1987)",
    "My mother thanks you. My father thanks you. My sister thanks you. And I thank you. – Yankee Doodle Dandy (1942)",
    "Nobody puts Baby in a corner. – Dirty Dancing (1987)",
    "I'll get you, my pretty, and your little dog too! – The Wizard of Oz (1939)",
    "I'm the king of the world! – Titanic (1997)"
]

len(top_100_movie_quotes)  # Confirm length is 100

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ------------------ Functions and response logic------------------

# ------------------ Bot Identity ------------------
def check_for_bot_identity(sentence):
    identity_patterns = [
        r'(who are you|what are you|tell me about yourself|your name|what can you do)'
    ]
    for pattern in identity_patterns:
        if re.search(pattern, sentence.lower()):
            return "I'm MovieVisionBot, your multimedia cinema assistant! I can recommend movies, explain plots, find genres, share quotes, and talk about directors. I can also show and process images with cool movie-themed effects!"
    return None

# ------------------ Naive Bayes Genre Classifier ------------------
movie_genres = []
for movie in imdb_top_movies:
    genre = movie["genre"][0].lower() if movie["genre"] else "unknown"
    sentence = f"{movie['title']} is a {genre} movie. {movie['plot']}"
    movie_genres.append((sentence, genre))

all_words = []
for text, label in movie_genres:
    words = word_tokenize(text.lower())
    all_words.extend(words)

word_features = list(set(all_words))

def document_features(text):
    words = word_tokenize(text.lower())
    return {word: (word in words) for word in word_features}

featuresets = [(document_features(text), label) for (text, label) in movie_genres]
classifier = nltk.NaiveBayesClassifier.train(featuresets)

def classify_movie_genre(text):
    feats = document_features(text)
    return classifier.classify(feats)

# ------------------ API Function for Movie Data ------------------
def get_movie_data(title):
    """Fetch movie data from OMDb API"""
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url).json()
        if response.get('Response') == 'True':
            return response
        else:
            return None
    except Exception as e:
        print(f"Error fetching movie data: {e}")
        return None


# ------------------ Image Processing Functions ------------------
def fetch_movie_poster(title):
    """Fetch movie poster from OMDb API"""
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url).json()

        if 'Poster' in response and response['Poster'] != 'N/A':
            poster_url = response['Poster']
            img_response = requests.get(poster_url)
            img = Image.open(BytesIO(img_response.content))

            # Save as preset image
            preset_path = f"{PRESET_DIR}/{title.replace(' ', '_')}.jpg"
            img.save(preset_path)

            return preset_path, response.get('Title', title)
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching poster: {e}")
        return None, None


def download_preset_posters():
    """Download posters for the top movies as preset images"""
    print("Downloading preset movie posters...")
    for movie in imdb_top_movies[:5]:  # Limit to 5 to avoid long startup time
        title = movie['title']
        path, _ = fetch_movie_poster(title)
        if path:
            print(f"Downloaded poster for {title}")
        else:
            print(f"Failed to download poster for {title}")


def load_user_image(image_path):
    """Load a user-provided image"""
    try:
        # Normalize path for OS
        image_path = os.path.normpath(image_path)

        if not os.path.exists(image_path):
            return None, f"Sorry, I couldn't find an image at {image_path}"

        img = Image.open(image_path)
        filename = os.path.basename(image_path)
        save_path = f"{USER_DIR}/{filename}"
        img.save(save_path)

        return save_path, f"Successfully loaded {filename}"
    except Exception as e:
        return None, f"Error loading image: {e}"


def apply_grayscale(image_path):
    """Convert image to grayscale"""
    try:
        img = Image.open(image_path)
        grayscale_img = img.convert('L')

        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        save_path = f"{PROCESSED_DIR}/{base_name}_grayscale.jpg"

        grayscale_img.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error applying grayscale: {e}")
        return None


def apply_edge_detection(image_path):
    """Apply edge detection to an image"""
    try:
        # Open image and convert to numpy array for OpenCV processing
        img = cv2.imread(image_path)
        edges = cv2.Canny(img, 100, 200)

        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        save_path = f"{PROCESSED_DIR}/{base_name}_edges.jpg"

        cv2.imwrite(save_path, edges)
        return save_path
    except Exception as e:
        print(f"Error applying edge detection: {e}")
        return None


def apply_cartoon_effect(image_path):
    """Apply cartoon effect to an image"""
    try:
        # Read the image
        img = cv2.imread(image_path)

        # Apply bilateral filter for smoothing but edge preserving
        color = cv2.bilateralFilter(img, 9, 250, 250)

        # Convert to grayscale and apply median blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)

        # Use adaptive thresholding to detect edges
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

        # Convert back to color format for OpenCV and combine the two
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges)

        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        save_path = f"{PROCESSED_DIR}/{base_name}_cartoon.jpg"

        cv2.imwrite(save_path, cartoon)
        return save_path
    except Exception as e:
        print(f"Error applying cartoon effect: {e}")
        return None


def apply_movie_poster_effect(image_path, title, year, rating):
    """
    Creative feature: Create a movie poster effect by adding:
    - Movie title at top
    - Year and rating at bottom
    - Cinematic letterbox effect
    - Film grain texture
    - Color grading
    """
    try:
        # Open image with PIL for text overlay
        img = Image.open(image_path)

        # Resize maintaining aspect ratio
        width, height = img.size
        ratio = width / height

        # Target poster dimensions (aspect ratio 2:3)
        if ratio > 0.67:  # Wider than a poster
            new_width = int(height * 0.67)
            left = (width - new_width) // 2
            img = img.crop((left, 0, left + new_width, height))
        else:  # Taller than a poster
            new_height = int(width / 0.67)
            top = (height - new_height) // 2
            img = img.crop((0, top, width, top + new_height))

        # Add letterbox effect (black bars)
        letterbox_height = int(img.height * 0.1)
        letterbox = Image.new('RGB', (img.width, img.height + 2 * letterbox_height), (0, 0, 0))
        letterbox.paste(img, (0, letterbox_height))
        img = letterbox

        # Apply color grading for cinematic effect
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)  # Slightly increase color saturation

        # Apply contrast adjustment
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)  # Slightly increase contrast

        # Add film grain effect
        grain = Image.effect_noise((img.width, img.height), 10)
        grain = grain.convert('RGB')
        img = Image.blend(img, grain, 0.05)  # 5% grain overlay

        # Prepare for text overlay
        draw = ImageDraw.Draw(img)

        # Try to load a font, use default if not available
        try:
            title_font = ImageFont.truetype("arial.ttf", 36)
            subtitle_font = ImageFont.truetype("arial.ttf", 18)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()

        # Add title text at top (centered)
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else img.width // 2
        title_position = ((img.width - title_width) // 2, letterbox_height // 2)

        # Add title with shadow effect
        draw.text((title_position[0] + 2, title_position[1] + 2), title, font=title_font, fill=(0, 0, 0))
        draw.text(title_position, title, font=title_font, fill=(255, 215, 0))  # Gold color

        # Add year and rating at bottom (centered)
        info_text = f"{year} • IMDb: {rating}/10"
        info_width = draw.textlength(info_text, font=subtitle_font) if hasattr(draw, 'textlength') else img.width // 3
        info_position = ((img.width - info_width) // 2, img.height - letterbox_height // 2)

        # Add info text with shadow
        draw.text((info_position[0] + 1, info_position[1] + 1), info_text, font=subtitle_font, fill=(0, 0, 0))
        draw.text(info_position, info_text, font=subtitle_font, fill=(255, 255, 255))

        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        save_path = f"{PROCESSED_DIR}/{base_name}_poster.jpg"

        img.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error applying movie poster effect: {e}")
        return None


def apply_vintage_film_effect(image_path):
    """
    Creative feature: Apply a vintage film look to the image with:
    - Sepia tone
    - Vignette
    - Film grain
    - Scratches
    """
    try:
        # Open image with PIL
        img = Image.open(image_path)

        # Convert to sepia tone
        img = img.convert('L')  # Convert to grayscale
        img = ImageOps.colorize(img, (60, 30, 15), (255, 240, 192))  # Sepia colorization

        # Add vignette effect
        width, height = img.size

        # Create a radial gradient mask
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)

        # Calculate parameters for vignette
        for i in range(min(width, height) // 4):
            ellipse_box = (i, i, width - i, height - i)
            intensity = 255 - int(255 * (i / (min(width, height) // 4)))
            draw.ellipse(ellipse_box, fill=intensity)

        # Apply vignette
        img = Image.composite(img, Image.new('RGB', img.size, (0, 0, 0)), mask)

        # Add film grain
        grain = Image.effect_noise((width, height), 15)
        grain = grain.convert('RGB')
        img = Image.blend(img, grain, 0.1)  # 10% grain overlay

        # Add scratches (vertical lines)
        draw = ImageDraw.Draw(img)
        num_scratches = 10
        for _ in range(num_scratches):
            x = random.randint(0, width)
            start_y = random.randint(0, height // 2)
            end_y = random.randint(start_y, height)
            draw.line((x, start_y, x, end_y), fill=(200, 200, 200), width=1)

        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        save_path = f"{PROCESSED_DIR}/{base_name}_vintage.jpg"

        img.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error applying vintage effect: {e}")
        return None


# ------------------ Text Processing Functions ------------------

def check_for_bot_identity(sentence):
    """Check if the user is asking about the bot's identity"""
    identity_patterns = [
        r'(who are you|what are you|tell me about yourself|your name|what can you do)'
    ]
    for pattern in identity_patterns:
        if re.search(pattern, sentence.lower()):
            return "I'm MovieVisionBot, your multimedia cinema assistant! I can recommend movies, provide info about films, show movie posters, and even process images with cool effects!"
    return None


def check_for_greeting(sentence):
    """Check if the input is a greeting and return a response"""
    sentence = sentence.lower().strip()
    # Match only if these are standalone words or at the beginning of the sentence
    for word in GREETING_KEYWORDS:
        if sentence == word or sentence.startswith(word + " "):
            return random.choice(GREETING_RESPONSES)
    return None


def check_for_farewell(sentence):
    """Check if the input is a farewell and return a response"""
    sentence = sentence.lower()
    for phrase in FAREWELL_KEYWORDS:
        if phrase in sentence:
            return random.choice(FAREWELL_RESPONSES)
    return None


# Movie recommendation patterns
def check_for_movie_recommendation(sentence):
    """Check if the user is asking for a movie recommendation"""
    movie_request_patterns = [
        r'recommend (a|some)? (movie|movies|film|films)',
        r'(suggest|give me) (a|some)? (movie|movies|film|films)',
        r'(what|which) (movie|film) (should|can) I (watch|see)',
        r'(any|some)? (good|best) (movie|movies|film|films) (to watch|for tonight|right now)?',
        r'(watch|see) (a|some)? (movie|film)',
        r'(any|some) (movie|film) recommendations',
        r'(movie|film) (please|suggestion|idea)?',
        r'i want to (watch|see) (a|some)? (movie|film)',
        r'what should I watch',
        r'suggest a (good)? film',
        r'movie (recommendation|suggestion)',
        r'film (recommendation|suggestion)',
        r'what to watch'
    ]

    for pattern in movie_request_patterns:
        if re.search(pattern, sentence.lower()):
            return recommend_movie()
    return None


def recommend_movie():
    """Recommend a movie from the IMDb Top 100 list"""
    movie = random.choice(imdb_top_movies)
    return f"I recommend '{movie['title']}' ({movie['year']}) directed by {movie['director']}. It has a rating of {movie['rating']}/10. Plot: {movie['plot']}"


# ------------------ Movie Info (using API) ------------------
def check_for_movie_info(sentence):
    """Check if the user is asking for information about a movie using API"""
    # Extract potential movie title from the sentence
    title_patterns = [
        r"tell me about (.+)",
        r"info(?:rmation)? (?:on|about) (.+)",
        r"what(?:'s| is) (.+) about",
        r"(.+?) movie info",
        r"(.+?) film info"
    ]

    for pattern in title_patterns:
        match = re.search(pattern, sentence.lower())
        if match:
            title = match.group(1).strip()
            movie_data = get_movie_data(title)

            if movie_data:
                return provide_movie_info_from_api(movie_data)
            else:
                return f"Sorry, I couldn't find information about '{title}'."

    # If no pattern matched, check if the sentence contains a movie title
    words = sentence.split()
    if len(words) >= 2:  # At least 2 words could be a movie title
        potential_title = ' '.join(words)
        movie_data = get_movie_data(potential_title)
        if movie_data:
            return provide_movie_info_from_api(movie_data)

    return None


def provide_movie_info_from_api(movie_data):
    """Format movie information from API data"""
    info = f"'{movie_data['Title']}' ({movie_data['Year']}) directed by {movie_data['Director']} "
    info += f"has an IMDb rating of {movie_data['imdbRating']}/10. "
    info += f"Genre: {movie_data['Genre']}. "
    info += f"Plot: {movie_data['Plot']}"
    return info


# ------------------ Movie Story (using API) ------------------
def check_for_story(sentence):
    """Check if the user is asking about a movie's plot using API"""
    sentence_lower = sentence.lower()

    # Patterns for general story requests (no specific movie)
    general_story_patterns = [
        r'^(tell|share)( me)? (a|your)? (movie )?(story|plot)$',
        r'^(movie|film) (summary|synopsis|plot)$',
        r'^give me a story$',
        r'^share a movie plot$'
    ]

    # Patterns for specific movie story requests
    specific_story_patterns = [
        r"(?:what's|what is) (.+) about",
        r"tell me (?:the )?story of (.+)",
        r"give me (.+) story",
    ]

    # First, check general story patterns (random movie from existing list)
    for pattern in general_story_patterns:
        if re.match(pattern, sentence_lower):
            movie = random.choice(imdb_top_movies)
            return f"Let me tell you the story of '{movie['title']}': {movie['plot']}"

    # Then, check specific movie story requests (using API)
    for pattern in specific_story_patterns:
        match = re.search(pattern, sentence_lower)
        if match:
            title = match.group(1).strip()
            movie_data = get_movie_data(title)

            if movie_data:
                return f"Here's the story of '{movie_data['Title']}': {movie_data['Plot']}"
            else:
                return f"Sorry, I couldn't find the plot for '{title}'."
    return None


# ------------------ Movie Quote ------------------
def check_for_movie_quote(sentence):
    """Check if the user is asking for a movie quote"""
    quote_patterns = [
        r'(famous|iconic)? (movie|film) (quote|line)',
        r'(say|share|tell me|give me) (a|some)? (movie|film)? (line|quote)',
        r'quote (from|of|in)? (a )?(film|movie)?',
        r'^quote$',
        r'^give me a quote$',
        r'^say a quote$'
    ]

    for pattern in quote_patterns:
        if re.search(pattern, sentence.lower()):
            return random.choice(top_100_movie_quotes) + " (from Top 100 Greatest Movie Quotes)"

    return None


# Movie sentiment patterns
def check_for_sentiment(sentence):
    """Check if the user is asking for an opinion about a movie using API"""
    sentence = sentence.lower()

    sentiment_patterns = [
        r"what do you think (about|of) (.+)",
        r"how do you (feel|think) about (.+)",
        r"is (.+?) (good|great|amazing|terrible|bad|awful)",
        r"would you recommend (.+)",
        r"do you (like|enjoy|recommend) (.+)"
    ]

    for pattern in sentiment_patterns:
        match = re.search(pattern, sentence)
        if match:
            # Extract the movie title from the appropriate group
            if "about" in pattern or "feel" in pattern:
                title = match.group(2).strip()
            else:
                title = match.group(1).strip()

            # Remove any trailing words like "any" that might be part of the question
            title = re.sub(r'\s+(any|good|great|bad)$', '', title)

            movie_data = get_movie_data(title)

            if movie_data:
                return movie_sentiment_from_api(movie_data)
            else:
                return f"Sorry, I couldn't find information about '{title}' to give you my opinion."
    return None


def movie_sentiment_from_api(movie_data):
    """Generate an opinion about a movie based on its data"""
    title = movie_data['Title']
    rating = float(movie_data['imdbRating']) if movie_data['imdbRating'] != 'N/A' else 0

    if rating >= 8.0:
        return f"I think '{title}' is a fantastic film! The {random.choice(['direction', 'acting', 'screenplay', 'cinematography'])} is particularly impressive."
    elif rating >= 7.0:
        return f"'{title}' is a pretty good movie with {random.choice(['solid performances', 'an engaging storyline', 'impressive visuals', 'memorable moments'])}."
    elif rating >= 5.0:
        return f"'{title}' is decent but has some {random.choice(['pacing issues', 'uneven moments', 'forgettable characters', 'plot holes'])}."
    else:
        return f"To be honest, '{title}' wasn't well received by critics or audiences. The {random.choice(['pacing', 'plot', 'character development', 'dialogue'])} could have been better."


# Movie director patterns
def check_for_director(sentence):
    """Check if the user is asking about a movie's director using API"""
    director_patterns = [
        r'who directed (.+)',
        r'director of (.+)',
        r'do you know who directed (.+)',
        r'who was the director of (.+)'
    ]

    for pattern in director_patterns:
        match = re.search(pattern, sentence.lower())
        if match:
            title = match.group(1).strip()
            # Remove question marks and other punctuation
            title = re.sub(r'[^\w\s]', '', title).strip()

            movie_data = get_movie_data(title)

            if movie_data:
                return f"'{movie_data['Title']}' was directed by {movie_data['Director']}."
            else:
                return f"Sorry, I couldn't find information about who directed '{title}'."
    return None


# ------------------ Movie Year (using API) ------------------
def check_for_year(sentence):
    """Check if the user is asking about a movie's release year using API"""
    year_patterns = [
        r'(?:when was|what year was|year of|release year of) (.+?) (?:released|come out|made)',
        r'what year (?:was|did) (.+?) (?:released|come out|made)',
        r'when did (.+?) (?:release|come out)',
        r'release year of (.+)'
    ]

    for pattern in year_patterns:
        match = re.search(pattern, sentence.lower())
        if match:
            title = match.group(1).strip()
            # Remove question marks and other punctuation
            title = re.sub(r'[^\w\s]', '', title).strip()

            movie_data = get_movie_data(title)

            if movie_data:
                return f"'{movie_data['Title']}' was released in {movie_data['Year']}."
            else:
                return f"Sorry, I couldn't find release year info for '{title}'."
    return None


# ------------------ Movie Genre (using API) ------------------
def check_for_genre(sentence):
    """Check if the user is asking about a movie's genre using API"""
    sentence_lower = sentence.lower()

    # First pattern: What genre is X?
    if "genre" in sentence_lower or "kind of movie" in sentence_lower:
        title_match = re.search(r'(what genre is|what kind of movie is|genre of) (.+)', sentence_lower)
        if title_match:
            title = title_match.group(2).strip()
            movie_data = get_movie_data(title)

            if movie_data:
                return f"'{movie_data['Title']}' is classified as: {movie_data['Genre']}."
            else:
                return f"Sorry, I don't have genre info for '{title}'."

    # Second pattern: Is X a Y?
    is_genre_match = re.search(r'is (.+?) an? (.+?)( movie)?(\?)?$', sentence_lower)
    if is_genre_match:
        title = is_genre_match.group(1).strip()
        genre = is_genre_match.group(2).strip()

        movie_data = get_movie_data(title)

        if movie_data:
            if genre.lower() in movie_data['Genre'].lower():
                return f"Yes, '{movie_data['Title']}' is a {genre} movie."
            else:
                return f"No, '{movie_data['Title']}' is not classified as a {genre} movie. It's classified as: {movie_data['Genre']}."
        else:
            return f"Sorry, I don't have genre info for '{title}'."

    return None


# ------------------ Show IMDb Top List ------------------
def check_for_list_request(sentence):
    """Check if the user is asking for the IMDb Top Movies list"""
    sentence_lower = sentence.lower()

    if ("imdb" in sentence_lower and ("top" in sentence_lower or "list" in sentence_lower)) or \
            ("top" in sentence_lower and ("movies" in sentence_lower or "films" in sentence_lower)) or \
            ("show" in sentence_lower and "list" in sentence_lower) or \
            ("print" in sentence_lower and ("movie" in sentence_lower or "list" in sentence_lower)):
        movies_list = "\n".join(
            [f"{i + 1}. {movie['title']} ({movie['year']}) - Rating: {movie['rating']}/10"
             for i, movie in enumerate(imdb_top_movies)])

        return f"Here’s the IMDb Top 100 Movies I'm using:\n{movies_list}"

    return None


# ------------------ Image Request Handlers ------------------
def check_for_poster_request(sentence):
    """Check if the user is asking for a movie poster"""
    poster_patterns = [
        r'(show|display|give me|load|get|see) (the )?(poster|image|picture) (of|for)? (.+)',
        r'(poster|image|picture) (of|for) (.+)'
    ]

    for pattern in poster_patterns:
        match = re.search(pattern, sentence.lower())
        if match:
            if 'poster' in pattern and len(match.groups()) >= 5:
                title = match.group(5).strip()
            else:
                title = match.group(3 if len(match.groups()) >= 3 else 2).strip()

            # See if we already have this poster
            preset_path = f"{PRESET_DIR}/{title.replace(' ', '_')}.jpg"

            if os.path.exists(preset_path):
                return preset_path, f"Here's the poster for '{title}'."

            # Try to find using the API
            path, actual_title = fetch_movie_poster(title)
            if path:
                return path, f"Here's the poster for '{actual_title}'."
            else:
                return None, f"Sorry, I couldn't find a poster for '{title}'."

    return None, None


def check_for_image_loading(sentence):
    """Check if the user wants to load an image"""
    pattern = r'(load|open|use|upload|process|show) (my |the |this |an? )?(image|picture|photo|file|pic)(?:\s+(?:from|at|called|named)?\s+(.+?))?(?:\.|$)'
    match = re.search(pattern, sentence.lower())

    if match:
        if match.group(4):  # There's a path or filename
            path = match.group(4).strip()
            # Remove quotes if present
            path = path.strip('"\'')
            return path
        else:
            return "prompt"  # Signal to prompt for a path

    return None


def check_for_image_effects(sentence, current_image_path):
    """Check if the user wants to apply an effect to the current image"""
    # First check if the sentence contains any effect-related words
    effect_words = ["grayscale", "black and white", "b&w", "edge", "cartoon",
                    "poster effect", "vintage", "apply effect", "effects"]

    is_effect_request = any(word in sentence.lower() for word in effect_words)

    # Only check for image if this is actually an effect request
    if is_effect_request and not current_image_path:
        return None, "Please load an image first before applying effects."

    # If not requesting an effect, return None, None
    if not is_effect_request:
        return None, None

    # Check for specific effects
    if re.search(r'(convert to|make|apply) (grayscale|black and white|b&w)',
                 sentence.lower()) or "grayscale" in sentence.lower():
        path = apply_grayscale(current_image_path)
        if path:
            return path, "Applied grayscale effect!"
        else:
            return None, "Failed to apply grayscale effect."

    elif re.search(r'(detect|find|show) edges', sentence.lower()) or "edge detection" in sentence.lower():
        path = apply_edge_detection(current_image_path)
        if path:
            return path, "Applied edge detection!"
        else:
            return None, "Failed to apply edge detection."

    elif re.search(r'(make|convert to|apply) cartoon', sentence.lower()) or "cartoon" in sentence.lower():
        path = apply_cartoon_effect(current_image_path)
        if path:
            return path, "Applied cartoon effect!"
        else:
            return None, "Failed to apply cartoon effect."

    elif re.search(r'(make|create|apply) (movie )?poster', sentence.lower()) or "poster effect" in sentence.lower():
        # Find a random movie to use for the poster effect if not specified
        movie = random.choice(imdb_top_movies)

        path = apply_movie_poster_effect(current_image_path, movie['title'], movie['year'], movie['rating'])
        if path:
            return path, f"Applied movie poster effect using '{movie['title']}' theme!"
        else:
            return None, "Failed to apply movie poster effect."

    elif re.search(r'(make|apply|convert to) vintage|old|retro film',
                   sentence.lower()) or "vintage" in sentence.lower():
        path = apply_vintage_film_effect(current_image_path)
        if path:
            return path, "Applied vintage film effect!"
        else:
            return None, "Failed to apply vintage film effect."

    elif re.search(r'(what|which|list) effects', sentence.lower()) or "effects" in sentence.lower():
        return None, "I can apply these effects to your image: grayscale, edge detection, cartoon, movie poster, vintage film. Just tell me which one you'd like!"

    return None, None

# ------------------ Main Loop ------------------
def main():
    print("🎬 Welcome to MovieVisionBot! Type 'exit' to quit.")
    print("I can recommend movies, show posters, and apply cool effects to images.")

    # Download preset posters on startup
    download_preset_posters()

    current_image_path = None


    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("MovieVisionBot:", random.choice(FAREWELL_RESPONSES))
            break

        user_input = user_input.strip()

        # Priority-based intent detection
        response = None

        # Check for basic intents first
        if check_for_farewell(user_input):
            response = check_for_farewell(user_input)
        elif check_for_greeting(user_input):
            response = check_for_greeting(user_input)
        elif check_for_bot_identity(user_input):
            response = check_for_bot_identity(user_input)
        # Then check for image-related commands (only if basic intents not matched)
        elif not response:
            # Check for poster request
            poster_path, poster_response = check_for_poster_request(user_input)
            if poster_path:
                current_image_path = poster_path
                print("MovieVisionBot:", poster_response)
                # Display the image
                img = cv2.imread(poster_path)
                cv2.imshow("Movie Poster", img)
                cv2.waitKey(1)  # Non-blocking wait
                continue

            # Check for image loading
            load_path = check_for_image_loading(user_input)
            if load_path:
                if load_path == "prompt":
                    print("MovieVisionBot: Please provide the path to your image:")
                    load_path = input("Path: ").strip('"\'')  # Strip quotes here
                    load_path = os.path.normpath(load_path)  # Normalize path for OS

                path, message = load_user_image(load_path)
                if path:
                    current_image_path = path
                    print("MovieVisionBot:", message)
                    # Display the image
                    img = cv2.imread(path)
                    cv2.imshow("User Image", img)
                    cv2.waitKey(1)
                else:
                    print("MovieVisionBot:", message)
                continue

            # Check for image effects
            effect_path, effect_response = check_for_image_effects(user_input, current_image_path)
            if effect_response and effect_path:
                current_image_path = effect_path
                print("MovieVisionBot:", effect_response)
                # Display the processed image
                img = cv2.imread(effect_path)
                cv2.imshow("Processed Image", img)
                cv2.waitKey(1)
                continue
            elif effect_response:
                print("MovieVisionBot:", effect_response)
                continue

            # Then check movie-specific intents
            elif check_for_list_request(user_input):
                response = check_for_list_request(user_input)
            elif check_for_movie_quote(user_input):
                response = check_for_movie_quote(user_input)
            elif check_for_director(user_input):
                response = check_for_director(user_input)
            elif check_for_year(user_input):
                response = check_for_year(user_input)
            elif check_for_genre(user_input):
                response = check_for_genre(user_input)
            elif check_for_story(user_input):
                response = check_for_story(user_input)
            elif check_for_movie_recommendation(user_input):
                response = check_for_movie_recommendation(user_input)
            elif check_for_sentiment(user_input):
                response = check_for_sentiment(user_input)
            elif check_for_movie_info(user_input):
                response = check_for_movie_info(user_input)

        # Print response if we have one
        if response:
            print("MovieVisionBot:", response)
        else:
            print(
                "MovieVisionBot: Hmm, I didn't quite catch that. Ask me about a movie, director, genre, quote, or try loading an image to apply cool effects!")


def handle_user_input(user_input):
    """Handle user input from web interface and return text response only"""
    user_input = user_input.strip()

    # Check for basic intents
    if check_for_farewell(user_input):
        return check_for_farewell(user_input)
    elif check_for_greeting(user_input):
        return check_for_greeting(user_input)
    elif check_for_bot_identity(user_input):
        return check_for_bot_identity(user_input)
    elif check_for_list_request(user_input):
        return "Here's the IMDb Top Movies list. I've formatted it in a table for better readability."
    elif check_for_movie_quote(user_input):
        return check_for_movie_quote(user_input)
    elif check_for_director(user_input):
        return check_for_director(user_input)
    elif check_for_year(user_input):
        return check_for_year(user_input)
    elif check_for_genre(user_input):
        return check_for_genre(user_input)
    elif check_for_story(user_input):
        return check_for_story(user_input)
    elif check_for_movie_recommendation(user_input):
        return check_for_movie_recommendation(user_input)
    elif check_for_sentiment(user_input):
        return check_for_sentiment(user_input)
    elif check_for_movie_info(user_input):
        return check_for_movie_info(user_input)
    elif "what effects" in user_input.lower() or "image effects" in user_input.lower():
        return "In the web version, I can show you movie posters, but advanced image processing features are only available in the desktop version."

    return "I'm not sure how to respond to that. Ask me about movies, directors, genres, or quotes!"

if __name__ == "__main__":
    main()