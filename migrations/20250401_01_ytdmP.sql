-- Create table vk_posts
-- depends: 
CREATE TABLE IF NOT EXISTS vk_posts (
    post_id INT PRIMARY KEY,
    text TEXT,
    date TIMESTAMP,
    likes INT,
    reposts INT,
    views INT
);
