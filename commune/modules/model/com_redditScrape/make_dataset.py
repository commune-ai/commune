from core.sql_access import SqlAccess
import pandas as pd

SQL = SqlAccess()
SQL.create_or_connect_db()
c = SQL.conn

all_data = pd.read_sql_query("""
SELECT *
FROM post p 
LEFT JOIN comment c 
    ON p.id = c.post_id
LEFT JOIN link l
	ON p.id = l.post_id;
""", c)

post = pd.read_sql_query("""
SELECT *
FROM post;
""", c)

comment = pd.read_sql_query("""
SELECT *
FROM comment;
""", c)

all_data.to_csv('data/post_comment_link_data.csv', columns=all_data.columns, index=False)
post.to_csv('data/post_data.csv', columns=post.columns, index=False)
comment.to_csv('data/comment_data.csv', columns=comment.columns, index=False)