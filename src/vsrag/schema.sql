-- Every statement uses IF NOT EXISTS, so running this file again changes nothing.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per complaint (keyed by ODINO) or recall campaign (keyed by CAMPNO).
-- Recalls have no crash, fire, injury or death fields, so those stay NULL for them.
CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,
    source      text NOT NULL CHECK (source IN ('complaint', 'recall')),
    title       text NOT NULL,
    body        text NOT NULL,
    search_text text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    received_on date,
    crash       boolean,
    fire        boolean,
    injured     integer,
    deaths      integer
);

CREATE INDEX IF NOT EXISTS documents_tsv_idx ON documents USING gin (tsv);

-- One complaint or recall can cover several vehicles and components.
CREATE TABLE IF NOT EXISTS document_vehicles (
    document_id text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    make        text NOT NULL,
    model       text NOT NULL,
    model_year  integer NOT NULL,
    component   text NOT NULL,
    PRIMARY KEY (document_id, make, model, model_year, component)
);

CREATE INDEX IF NOT EXISTS document_vehicles_vehicle_idx
    ON document_vehicles (make, model, model_year);

-- One row per document per embedding variant, so variants can be compared side by side.
-- Each variant gets its own HNSW index when its embeddings are built (see embed.py).
CREATE TABLE IF NOT EXISTS embeddings (
    document_id text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    variant     text NOT NULL,
    embedding   vector(384) NOT NULL,
    PRIMARY KEY (document_id, variant)
);
