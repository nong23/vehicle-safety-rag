"""Find and read NHTSA documents, to write and verify test-set answers.

    python -m evals.lookup airbag deploy --model TELLURIDE      documents containing every word
    python -m evals.lookup --model TELLURIDE --component "AIR BAGS"     no words: browse by filters
    python -m evals.lookup --id complaint:11331228               one document's full text

This deliberately doesn't use the app's own search. Picking "relevant" documents with the
system you're testing would make its recall look better than it is, so this is a plain
text match: every word must appear somewhere in the document (case doesn't matter).
"""

import argparse

from vsrag.db import get_conn

PREVIEW_CHARS = 220


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("words", nargs="*", help="words that must all appear in the text")
    parser.add_argument("--id", help="print this document's full text")
    parser.add_argument("--make")
    parser.add_argument("--model")
    parser.add_argument("--year", type=int)
    parser.add_argument("--component", help="part of NHTSA's component name, e.g. 'AIR BAGS'")
    parser.add_argument("--source", choices=["complaint", "recall"])
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    with get_conn() as conn:
        if args.id:
            row = conn.execute(
                "SELECT id, title, received_on, body FROM documents WHERE id = %s", (args.id,)
            ).fetchone()
            if row is None:
                print(f"No document {args.id}")
                return
            vehicles = conn.execute(
                "SELECT DISTINCT model_year, make, model, component FROM document_vehicles"
                " WHERE document_id = %s ORDER BY 1, 2, 3, 4",
                (args.id,),
            ).fetchall()
            print(f"{row[0]}  |  {row[1]}  |  {row[2]}")
            for year, make, model, component in vehicles:
                print(f"  vehicle: {year} {make} {model}  |  {component}")
            print(f"\n{row[3]}")
            return

        conditions = ["TRUE"]
        params: list[object] = []
        for word in args.words:
            conditions.append("d.body ILIKE %s")  # ILIKE = LIKE, ignoring upper/lower case
            params.append(f"%{word}%")
        if args.source:
            conditions.append("d.source = %s")
            params.append(args.source)
        vehicle = []
        for column, value in (
            ("make", args.make),
            ("model", args.model),
            ("model_year", args.year),
        ):
            if value is not None:
                vehicle.append(f"v.{column} = %s")
                params.append(value.upper() if isinstance(value, str) else value)
        if args.component:
            vehicle.append("v.component ILIKE %s")
            params.append(f"%{args.component}%")
        if vehicle:
            conditions.append(
                "EXISTS (SELECT 1 FROM document_vehicles v WHERE v.document_id = d.id AND "
                + " AND ".join(vehicle)
                + ")"
            )
        where = " AND ".join(conditions)
        total = conn.execute(f"SELECT count(*) FROM documents d WHERE {where}", params).fetchone()
        rows = conn.execute(
            f"SELECT d.id, d.title, d.body FROM documents d WHERE {where}"
            " ORDER BY d.source DESC, d.id LIMIT %s",  # recalls first: there are fewer of them
            [*params, args.limit],
        ).fetchall()

    print(f"{total[0] if total else 0} matching documents (showing {len(rows)})\n")
    for doc_id, title, body in rows:
        preview = " ".join(body.split())[:PREVIEW_CHARS]
        print(f"{doc_id}  |  {title}\n    {preview}\n")


if __name__ == "__main__":
    main()
