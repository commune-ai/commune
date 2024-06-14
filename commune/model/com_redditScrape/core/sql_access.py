import sqlite3
import os

class SqlAccess():
    def __init__(self,
                 name = 'reddit_data'):
        self.conn = None
        self.db_name = name
        self.last_post_id = 0
    
    def create_or_connect_db(self, erase_first=False):
        if(erase_first):
            try:
                os.remove(('{0}.db').format(self.db_name))
            except Exception:
                pass
        
        # Creates and/or connects to SQLite database
        self.conn = sqlite3.connect(('data/{0}.db').format(self.db_name))
        c = self.conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS post
                     (
                     id             INTEGER     PRIMARY KEY     AUTOINCREMENT,
                     url            varchar     NOT NULL,
                     url_id         varchar     NOT NULL,
                     url_title      varchar     NOT NULL,
                     author         varchar     NOT NULL,
                     upvote_ratio   uint8       NOT NULL,
                     score          int         NOT NULL,
                     time_created   datetime    NOT NULL,
                     num_gold       int,
                     num_comments   int,
                     category       varchar,
                     text           varchar,
                     main_link      varchar,
                     flairs         int
                     )
                     ''')
        # score is varchar, because sometimes fourth span is not a score,
        # could be username, original poster or something else
        c.execute('''CREATE TABLE IF NOT EXISTS comment
                     (
                     id             INTEGER     PRIMARY KEY     AUTOINCREMENT,
                     post_id        int         NOT NULL,
                     comment_id     varchar     NOT NULL,
                     score          varchar     NOT NULL,
                     depth          int         NOT NULL,
                     next           varchar,
                     previous       varchar,
                     comment_author varchar,
                     text           varchar,
                     FOREIGN KEY (post_id) REFERENCES post (id)
                     )
                     ''')
                     
        c.execute('''CREATE TABLE IF NOT EXISTS link
                     (
                     id             INTEGER     PRIMARY KEY     AUTOINCREMENT,
                     post_id        int         NOT NULL,
                     link           varchar     NOT NULL,
                     FOREIGN KEY (post_id) REFERENCES post (id)
                     )
                     ''')
        
    def save_changes(self):
        self.conn.commit()
        self.conn.close()
        
    def _question_mark_creator(self,
                               n_question_marks):
        final_string = ''
        
        for i in range(n_question_marks):
            final_string += '?,'
            
        final_string += '?'
        
        return final_string

    def insert(self,
               table,
               data):
        '''
            A general function for inserting data into tables.
            Insert is made to be single insert only, such that the
            autoincremented ids can be retrieved after insertion. This would
            not be possible with .executemany() instead of .execute()
        '''
        c = self.conn.cursor()
        
        # Get the column names of the table we are trying to insert into
        cols = c.execute(('''
                        PRAGMA table_info({0})
                        ''').format(table))
        
        # Get the number of columns
        num_cols = sum([1 for i in cols]) - 1
        
        # Generate question marks for VALUES insertion
        question_marks = self._question_mark_creator(num_cols)
        
        if table == 'post':
            c.execute(('''INSERT INTO {0}
                          VALUES ({1})'''
                      ).format(table, question_marks), data)
            
            self.last_post_id = c.lastrowid
            
        elif (table == 'comment' or table == 'link') \
             and data != None and data != []:
            # setting post_id to the last post id, inserted in the post insert
            for table_data in data:
                table_data[1] = self.last_post_id
                c.execute(('''INSERT INTO {0}
                              VALUES ({1})'''
                          ).format(table, question_marks), table_data)
    