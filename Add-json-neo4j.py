from neo4j import GraphDatabase
import json

# Configuration de Neo4j
uri = ""
username = ""
password = ""

driver = GraphDatabase.driver(uri, auth=(username, password))

# Charger le fichier JSON
with open("ALL.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Fonction pour créer les nœuds et les relations
def create_graph(tx, node1, node2, relationship, description):
    label1 = node1["label"].replace(" ", "_")
    label2 = node2["label"].replace(" ", "_")
    rel_type = relationship.replace(" ", "_")

    # Sécurité : vérifier que les labels ne sont pas vides
    if not label1 or not label2:
        raise ValueError(f"Empty label found: label1='{label1}', label2='{label2}'")

    query = f"""
    MERGE (a:`{label1}` {{name: $name1}})
    ON CREATE SET a.source = $source1, a.link = $link1

    MERGE (b:`{label2}` {{name: $name2}})
    ON CREATE SET b.source = $source2, b.link = $link2

    MERGE (a)-[r:`{rel_type}` {{description: $description}}]->(b)
    """

    tx.run(query, 
           name1=node1["name"], source1=node1["source"], link1=node1["link"],
           name2=node2["name"], source2=node2["source"], link2=node2["link"],
           description=description)


# Connexion à Neo4j et ajout des données
with driver.session() as session:
    for i, entry in enumerate(data):
        try:
            node1 = entry["node_1"]
            node2 = entry["node_2"]
            relationship = entry["relationship"]
            description = entry["description"]
            session.execute_write(create_graph, node1, node2, relationship, description)
        except Exception as e:
            print(f"Erreur à l'entrée {i} : {e}")

driver.close()
