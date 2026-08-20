import sqlite3

c = sqlite3.connect("/home/swdc-kjsse/proxy/data/database.sqlite")
print("tables:", c.execute("select name from sqlite_master").fetchall())
print("---hosts---")
try:
    rows = c.execute(
        "select id, domain_names, forward_host, forward_port, ssl_forced from proxy_host"
    ).fetchall()
    for r in rows:
        print(r)
except Exception as e:
    print("proxy_host err", e)
    for row in c.execute("select name from sqlite_master where type='table'"):
        print(row)
