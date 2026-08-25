using Microsoft.Data.Sqlite;

namespace NeuralLab.Server;

/// <summary>
/// The one place the schema is defined and the one place a connection is opened from. No ORM —
/// one table and five hand-written queries do not need one, and a project that hand-writes its
/// own gradients rather than reach for a library (invariant 6) is not about to reach for a heavy
/// one here either. A fresh <see cref="SqliteConnection"/> per call rather than a shared, pooled
/// one — this is a personal project's local database, not a multi-tenant service, and a fresh
/// connection per request removes any question of connections being shared across requests.
/// </summary>
public static class Db
{
    public static string ConnectionString { get; private set; } = "Data Source=neurallab.db";

    public static void Init(string dataDirectory)
    {
        Directory.CreateDirectory(dataDirectory);
        var path = Path.Combine(dataDirectory, "neurallab.db");
        ConnectionString = $"Data Source={path}";

        using var conn = Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS runs (
                id            TEXT PRIMARY KEY,
                owner_id      TEXT NOT NULL,
                title         TEXT,
                net           TEXT NOT NULL,
                dataset       TEXT NOT NULL,
                config        TEXT NOT NULL,
                final_metrics TEXT NOT NULL,
                final_loss    REAL NOT NULL,
                share_token   TEXT,
                created_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs(owner_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_share_token ON runs(share_token)
                WHERE share_token IS NOT NULL;
            """;
        cmd.ExecuteNonQuery();
    }

    public static SqliteConnection Open()
    {
        var conn = new SqliteConnection(ConnectionString);
        conn.Open();
        return conn;
    }
}
