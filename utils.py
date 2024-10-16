from py2neo import Graph
import credentials

def get_graph():
    return Graph(credentials.NEO4J_URI, auth=(credentials.NEO4J_USERNAME, credentials.NEO4J_PASSWORD))

def fetch_user_data(user_id):
    graph = get_graph()
    query = f'MATCH (u:User {{id: {user_id}}}) RETURN u'
    return graph.run(query).data()

def fetch_movie_data(movie_id):
    graph = get_graph()
    query = f'MATCH (m:Movie {{id: {movie_id}}}) RETURN m'
    return graph.run(query).data()
