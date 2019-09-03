###############################################################################################
# Database
# Input:
#     connection : dict(), connection information - host, database, user, password, port
# Output:
#     cursor : Returns cursor for the connection
###############################################################################################
class Database():
    def __init__(self,connection):
        self.conn = None
        self.tries = 0
        while self.tries < 10:
            try:
                self.conn = psycopg2.connect(**connection)
                if self.conn is not None:
                    print('Connected')
                    break
            except:
                print('Unable to connect. Try again')
                self.tries += 1

    def cursor(self):
        return self.conn.cursor()
    
    def close_cursor(self):
        self.conn.cursor.close()
        
    def close(self):
        self.conn.close()
