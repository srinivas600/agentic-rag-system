"""Verify seeded data in both SQL and Vector databases."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings


async def main():
    from app.models.database import async_session_factory
    from sqlalchemy import text as sql_text

    print("=" * 60)
    print("  Data Verification Report")
    print("=" * 60)

    # ── SQL Database ─────────────────────────────────────────
    async with async_session_factory() as session:
        # Products
        result = await session.execute(sql_text("SELECT COUNT(*) FROM products"))
        product_count = result.scalar()

        result = await session.execute(sql_text(
            "SELECT category, COUNT(*), ROUND(AVG(price), 2), ROUND(SUM(price), 2) "
            "FROM products GROUP BY category ORDER BY category"
        ))
        categories = result.fetchall()

        print(f"\n  PRODUCTS ({product_count} total)")
        print(f"  {'Category':<15} {'Count':>6} {'Avg Price':>12} {'Total Value':>14}")
        print(f"  {'-'*15} {'-'*6} {'-'*12} {'-'*14}")
        for cat, count, avg_price, total in categories:
            print(f"  {cat:<15} {count:>6} ${avg_price:>10.2f} ${total:>12.2f}")

        # Documents
        result = await session.execute(sql_text("SELECT COUNT(*) FROM documents"))
        doc_count = result.scalar()

        result = await session.execute(sql_text(
            "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type ORDER BY doc_type"
        ))
        doc_types = result.fetchall()

        result = await session.execute(sql_text(
            "SELECT COUNT(*) FROM documents WHERE parent_chunk_id IS NULL AND token_count > 256"
        ))
        parent_count = result.scalar()

        result = await session.execute(sql_text(
            "SELECT COUNT(*) FROM documents WHERE parent_chunk_id IS NOT NULL"
        ))
        child_count = result.scalar()

        print(f"\n  DOCUMENTS ({doc_count} total chunks — {parent_count} parent, {child_count} child)")
        print(f"  {'Doc Type':<15} {'Chunks':>8}")
        print(f"  {'-'*15} {'-'*8}")
        for dtype, count in doc_types:
            print(f"  {dtype:<15} {count:>8}")

        # Sample titles
        result = await session.execute(sql_text(
            "SELECT DISTINCT title FROM documents ORDER BY title"
        ))
        titles = [row[0] for row in result.fetchall()]
        print(f"\n  UNIQUE DOCUMENT TITLES ({len(titles)}):")
        for t in titles:
            print(f"    - {t}")

    # ── Pinecone Vector DB ───────────────────────────────────
    print(f"\n  PINECONE INDEX: {settings.pinecone_index_name}")
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        stats = index.describe_index_stats()
        print(f"  Total vectors: {stats.total_vector_count}")
        print(f"  Dimension: {stats.dimension}")
        if stats.namespaces:
            for ns, ns_stats in stats.namespaces.items():
                ns_label = ns if ns else "(default)"
                print(f"  Namespace '{ns_label}': {ns_stats.vector_count} vectors")
    except Exception as e:
        print(f"  Error checking Pinecone: {e}")

    print(f"\n{'='*60}")
    print("  Verification Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
