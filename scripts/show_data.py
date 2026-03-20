"""Show all data currently stored in the databases."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.models.database import async_session_factory
from sqlalchemy import text


async def main():
    async with async_session_factory() as s:
        # Products
        r = await s.execute(text(
            "SELECT name, category, price, inventory FROM products ORDER BY category, name"
        ))
        rows = r.fetchall()
        print("=" * 70)
        print(f"  PRODUCTS ({len(rows)} items)")
        print("=" * 70)
        current_cat = ""
        for name, cat, price, inv in rows:
            if cat != current_cat:
                current_cat = cat
                print(f"\n  [{cat}]")
            print(f"    {name:<45} ${price:>9.2f}  (qty: {inv})")

        # Documents
        r = await s.execute(text(
            "SELECT DISTINCT title, doc_type FROM documents "
            "WHERE parent_chunk_id IS NULL AND token_count > 200 "
            "ORDER BY doc_type, title"
        ))
        docs = r.fetchall()
        sep = "=" * 70
        print(f"\n\n{sep}")
        print(f"  DOCUMENTS ({len(docs)} titles)")
        print(sep)
        current_type = ""
        for title, dtype in docs:
            if dtype != current_type:
                current_type = dtype
                print(f"\n  [{dtype}]")
            print(f"    {title}")

        # Stats
        r = await s.execute(text("SELECT COUNT(*) FROM documents"))
        total_chunks = r.scalar()
        r = await s.execute(text(
            "SELECT COUNT(*) FROM documents WHERE parent_chunk_id IS NOT NULL"
        ))
        child = r.scalar()
        r = await s.execute(text("SELECT COUNT(*) FROM products"))
        prod_count = r.scalar()
        print(f"\n\n{sep}")
        print("  SUMMARY")
        print(sep)
        print(f"  Products:        {prod_count}")
        print(f"  Document chunks: {total_chunks} ({total_chunks - child} parent + {child} child)")
        print(f"  Pinecone vectors: {child} (child chunks embedded)")
        print(sep)


asyncio.run(main())
