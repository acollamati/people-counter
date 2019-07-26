from elasticsearch import Elasticsearch

es = Elasticsearch([{'host':'localhost','port':9200}])

faces = {}
faces_es = []

business_id = "cafe124"
shop_id = "sottocorno17"

with open("face_detected.log") as f:
    for l in f.readlines():
        l = l.split(",")
        if faces.get(int(l[1])) is not None:
            previous_visit = (faces.get(int(l[1]))[-1]["visit_date"] - int(l[0])) / 86400000
            faces[int(l[1])].append({"business_id": business_id, "shop_id": shop_id,
                                     "customer": int(l[1]), "visit_date": int(l[0]),
                                     "camera": l[6].strip(), "previous_visit": previous_visit})
        else:
            faces[int(l[1])] = [{"business_id": business_id, "shop_id": shop_id,
                                 "customer": int(l[1]), "visit_date": int(l[0]),
                                 "camera": l[6].strip(), "previous_visit": 0}]

        faces_es.append(faces[int(l[1])][-1])
            
mappings = {
  "mappings": {
    "properties": {
        "shop_id":  {"type": "keyword"},
        "business_id":  {"type": "keyword"},
        "customer": {"type": "keyword"},
        "camera": {"type": "keyword"},
        "previous_visit":      {"type": "integer"},
        "visit_date":  {
            "type":   "date",
            "format": "strict_date_optional_time||epoch_millis"
      }
    }
  }
}

index_name = "people-counter"

es.indices.delete(index=index_name)
es.indices.create(index=index_name, body=mappings)

for e in faces_es:
    es.index(index=index_name, body=e)

# es.get(index=index_name, body={'query':{}})
