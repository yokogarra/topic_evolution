
// Load nodes
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/55347/OneDrive/Desktop/topic_evolution/ten_out/nodes.csv' AS row
MERGE (t:Topic {id: row.node_id})
SET t.time_id = toInteger(row.time_id),
    t.time_from = toInteger(row.time_from),
    t.time_to   = toInteger(row.time_to),
    t.tid       = toInteger(row.tid),
    t.size      = toInteger(row.size),
    t.coherence = toFloat(row.coherence),
    t.label     = row.label,
    t.top_terms = row.top_terms,
    t.mean_score = CASE WHEN row.mean_score = '' THEN null ELSE toFloat(row.mean_score) END,
    t.sum_fav    = CASE WHEN row.sum_fav = '' THEN null ELSE toFloat(row.sum_fav) END;

// Load edges
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/55347/OneDrive/Desktop/topic_evolution/ten_out/edges.csv' AS row
MATCH (a:Topic {id: row.source})
MATCH (b:Topic {id: row.target})
MERGE (a)-[e:EVOLVE {kind: row.kind}]->(b)
SET e.weight = toFloat(row.weight),
    e.time_from = toInteger(row.time_from),
    e.time_to   = toInteger(row.time_to);
