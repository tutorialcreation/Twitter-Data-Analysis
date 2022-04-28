USE twitter;
CREATE TABLE userData(
    id INT AUTO_INCREMENT,
    created_at DATETIME,
    source VARCHAR(255),
    original_text TEXT,
    polarity FLOAT,
    subjectivity FLOAT,
    lang VARCHAR(50),
    favorite_count INT,
    retweet_count INT,
    original_author VARCHAR(255),
    followers_count INT,
    friends_count INT,
    possibly_sensitive VARCHAR(50),
    hashtags VARCHAR(255),
    user_mentions VARCHAR(50),
    place VARCHAR(50),
    PRIMARY KEY(id)
);