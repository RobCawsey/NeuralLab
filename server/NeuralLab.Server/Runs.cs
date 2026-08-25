using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace NeuralLab.Server;

public sealed record SaveRunRequest(string? Title, string Net, string Dataset, string Config, JsonElement FinalMetrics, double FinalLoss);
public sealed record RunSummary(string Id, string? Title, string Net, string Dataset, double FinalLoss, string CreatedAt);
public sealed record RunDetail(string Id, string? Title, string Net, string Dataset, string Config, JsonElement FinalMetrics, string CreatedAt);
public sealed record ShareResult(string Token);

/// <summary>
/// Storage only — every field here is opaque to the server except the handful of columns "list
/// mine" needs to sort and filter by. Nothing is uploaded that could not be recomputed:
/// <c>Config</c> is the same query string <c>writeUrl</c>/<c>writeSomUrl</c> already produce for
/// the address bar (§8), so reopening a run is re-reading a URL, not fetching the 354 floats §10
/// says explicitly never leave the browser. <c>FinalMetrics</c> is likewise opaque JSON the
/// client both writes and reads — an MLP run's shape (train/val loss and accuracy) and a SOM
/// run's (QE/TE) are different, and the server has no reason to know either one's fields, only to
/// hand the same bytes back.
/// </summary>
public static class Runs
{
    public static string Save(string ownerId, SaveRunRequest req)
    {
        var id = Guid.NewGuid().ToString("N");
        using var conn = Db.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO runs (id, owner_id, title, net, dataset, config, final_metrics, final_loss, created_at)
            VALUES ($id, $owner, $title, $net, $dataset, $config, $metrics, $loss, $created)
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$owner", ownerId);
        cmd.Parameters.AddWithValue("$title", (object?)req.Title ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$net", req.Net);
        cmd.Parameters.AddWithValue("$dataset", req.Dataset);
        cmd.Parameters.AddWithValue("$config", req.Config);
        cmd.Parameters.AddWithValue("$metrics", req.FinalMetrics.GetRawText());
        cmd.Parameters.AddWithValue("$loss", req.FinalLoss);
        cmd.Parameters.AddWithValue("$created", DateTimeOffset.UtcNow.ToString("O"));
        cmd.ExecuteNonQuery();
        return id;
    }

    public static List<RunSummary> ListMine(string ownerId)
    {
        using var conn = Db.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = """
            SELECT id, title, net, dataset, final_loss, created_at
            FROM runs WHERE owner_id = $owner ORDER BY created_at DESC
            """;
        cmd.Parameters.AddWithValue("$owner", ownerId);
        using var reader = cmd.ExecuteReader();
        var result = new List<RunSummary>();
        while (reader.Read())
        {
            result.Add(new RunSummary(
                reader.GetString(0),
                reader.IsDBNull(1) ? null : reader.GetString(1),
                reader.GetString(2),
                reader.GetString(3),
                reader.GetDouble(4),
                reader.GetString(5)));
        }
        return result;
    }

    /// <summary>Owner-scoped reopen — the plain-id path only a run's own owner can use.</summary>
    public static RunDetail? GetOwned(string id, string ownerId)
    {
        using var conn = Db.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = """
            SELECT id, title, net, dataset, config, final_metrics, created_at
            FROM runs WHERE id = $id AND owner_id = $owner
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$owner", ownerId);
        using var reader = cmd.ExecuteReader();
        return reader.Read() ? ReadDetail(reader) : null;
    }

    /// <summary>The public path a share token grants — deliberately no owner check.</summary>
    public static RunDetail? GetShared(string token)
    {
        using var conn = Db.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = """
            SELECT id, title, net, dataset, config, final_metrics, created_at
            FROM runs WHERE share_token = $token
            """;
        cmd.Parameters.AddWithValue("$token", token);
        using var reader = cmd.ExecuteReader();
        return reader.Read() ? ReadDetail(reader) : null;
    }

    /// <summary>
    /// Idempotent — sharing an already-shared run returns the same token rather than minting a
    /// second one, so pressing Share twice does not invalidate a link already handed out. Returns
    /// null for a run that does not exist or is not owned by <paramref name="ownerId"/>, which the
    /// endpoint turns into 404 either way — a reader probing other people's run ids learns nothing
    /// from the difference between "wrong id" and "not yours".
    /// </summary>
    public static string? Share(string id, string ownerId)
    {
        using var conn = Db.Open();

        using (var check = conn.CreateCommand())
        {
            check.CommandText = "SELECT share_token FROM runs WHERE id = $id AND owner_id = $owner";
            check.Parameters.AddWithValue("$id", id);
            check.Parameters.AddWithValue("$owner", ownerId);
            var existing = check.ExecuteScalar();
            if (existing is null) return null;
            if (existing is string s) return s;
        }

        var token = Guid.NewGuid().ToString("N");
        using var update = conn.CreateCommand();
        update.CommandText = "UPDATE runs SET share_token = $token WHERE id = $id AND owner_id = $owner";
        update.Parameters.AddWithValue("$token", token);
        update.Parameters.AddWithValue("$id", id);
        update.Parameters.AddWithValue("$owner", ownerId);
        update.ExecuteNonQuery();
        return token;
    }

    private static RunDetail ReadDetail(SqliteDataReader reader) => new(
        reader.GetString(0),
        reader.IsDBNull(1) ? null : reader.GetString(1),
        reader.GetString(2),
        reader.GetString(3),
        reader.GetString(4),
        JsonDocument.Parse(reader.GetString(5)).RootElement,
        reader.GetString(6));
}
