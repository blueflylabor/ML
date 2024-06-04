import multiprocessing as mp

def foo(conn):
    conn.send(['hello', None, 42])
    conn.close()
if __name__ == '__main__':
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=foo, args=(child_conn,))
    p.start()
    print(parent_conn.recv())
    p.join()