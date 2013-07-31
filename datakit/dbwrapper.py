# Copyright (c) 2013 Andrew Werner and Anthony DeGangi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import MySQLdb

def get_db_connection(**kwargs):
    return MySQLdb.connect(**kwargs)

def get_db_cursor(db, unbuffered=True, dicts=True):
    if unbuffered and dicts:
        cursor = MySQLdb.cursors.SSDictCursor
    elif not unbuffered and dicts:
        cursor = MySQLdb.cursors.DictCursor
    elif unbuffered and not dicts:
        cursor = MySQLdb.cursors.SSCursor
    elif not unbuffered and not dicts:
        cursor = MySQLdb.cursors.Cursor
    return db.cursor(cursor)


class DbConn(object):
    def __init__(self, host, user, password, db, connect_args=None,
                 unbuffered=True, dicts=True, reconnect=True):
        if connect_args is None:
            connect_args = {}
        self.connection = get_db_connection(host=host, user=user, 
                                            passwd=password, 
                                            db=db, **connect_args)
        self.cursor = get_db_cursor(self.connection, unbuffered, dicts)
        self.unbuffered = unbuffered
        self.dicts = dicts
        self.reconnect = reconnect

    def execute(self, sql, args=None):
        sql = self._format_sql(sql, args)
        if self.reconnect == True:
            self.connection.ping(True)

        self.cursor.execute(sql)

        # try:
            # self.cursor.execute(sql)
        # except (AttributeError, MySQLdb.OperationalError):
            # self.reconnect(sql)

        return self.cursor

    def _format_sql(self, sql, args=None):
        if args:
            try:
                stmt = sql % args
            except:
                stmt = "<unprintable sql>"
        else:
            stmt = sql
        return stmt

    def reconnect(self, sql):
        try:
            self.close()
        except:
            pass

        self.connection = get_db_connection(connect_args)
        self.cursor = get_db_cursor(self.connection, self.unbuffered, 
                                    self.dicts)
        self.cursor.execute(sql)
        return

    def close(self):
        self.cursor.close()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


if __name__ == '__main__':
    with DbConn() as db:
        # pull data from headliner DE
        t = db.execute("SELECT * FROM users limit 3")
        print t.fetchone()
        print 1
        print t.fetchone()
        print 2
        print t.fetchone()
        print 3
        #you need to clear out the query to avoid an exception
        print t.fetchone()
        print 4

        # get all rows
        t = db.execute("SELECT * FROM promotions limit 3")
        rows = t.fetchall()
        print rows
        print "all 3"
