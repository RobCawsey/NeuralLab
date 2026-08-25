using System.Text.Json;
using NeuralLab.Server;
using Xunit;

namespace NeuralLab.Server.Tests;

/// <summary>
/// Against a real, temporary SQLite file rather than a mock — the thing worth trusting here is
/// the SQL, and a mock would just be a second, hand-written copy of what the SQL is supposed to
/// do. Every test gets its own directory, since xUnit gives each test method a fresh class
/// instance and the constructor re-points <see cref="Db"/>'s single, static connection string.
/// </summary>
public class RunsTests
{
    public RunsTests()
    {
        var dir = Path.Combine(Path.GetTempPath(), "neurallab-tests-" + Guid.NewGuid().ToString("N"));
        Db.Init(dir);
    }

    private static SaveRunRequest SampleMlp(string title = "two moons · 2-8-8-2") => new(
        Title: title,
        Net: "mlp",
        Dataset: "moons",
        Config: "net=mlp&data=moons&arch=8-8&steps=400",
        FinalMetrics: JsonSerializer.SerializeToElement(new { trainLoss = 0.1007, valAccuracy = 0.9861 }),
        FinalLoss: 0.1007);

    [Fact]
    public void Save_then_GetOwned_round_trips_every_field()
    {
        var id = Runs.Save("owner-a", SampleMlp());

        var detail = Runs.GetOwned(id, "owner-a");

        Assert.NotNull(detail);
        Assert.Equal(id, detail!.Id);
        Assert.Equal("two moons · 2-8-8-2", detail.Title);
        Assert.Equal("mlp", detail.Net);
        Assert.Equal("moons", detail.Dataset);
        Assert.Equal("net=mlp&data=moons&arch=8-8&steps=400", detail.Config);
        Assert.Equal(0.1007, detail.FinalMetrics.GetProperty("trainLoss").GetDouble());
        Assert.Equal(0.9861, detail.FinalMetrics.GetProperty("valAccuracy").GetDouble());
    }

    [Fact]
    public void GetOwned_refuses_a_different_owner_even_with_the_right_id()
    {
        var id = Runs.Save("owner-a", SampleMlp());

        Assert.Null(Runs.GetOwned(id, "owner-b"));
    }

    [Fact]
    public void ListMine_returns_only_that_owners_runs_newest_first()
    {
        var first = Runs.Save("owner-a", SampleMlp("first"));
        Thread.Sleep(5); // created_at has second-ish resolution in some environments; keep the order unambiguous
        var second = Runs.Save("owner-a", SampleMlp("second"));
        Runs.Save("owner-b", SampleMlp("not mine"));

        var mine = Runs.ListMine("owner-a");

        Assert.Equal(2, mine.Count);
        Assert.Equal(second, mine[0].Id);
        Assert.Equal(first, mine[1].Id);
        Assert.All(mine, r => Assert.NotEqual("not mine", r.Title));
    }

    [Fact]
    public void Share_is_idempotent_and_scoped_to_the_owner()
    {
        var id = Runs.Save("owner-a", SampleMlp());

        var token1 = Runs.Share(id, "owner-a");
        var token2 = Runs.Share(id, "owner-a");

        Assert.NotNull(token1);
        Assert.Equal(token1, token2); // same link handed out twice, not silently invalidated

        Assert.Null(Runs.Share(id, "owner-b")); // cannot mint a link for a run you do not own
    }

    [Fact]
    public void GetShared_works_by_token_regardless_of_owner_and_rejects_a_wrong_token()
    {
        var id = Runs.Save("owner-a", SampleMlp());
        var token = Runs.Share(id, "owner-a")!;

        var byToken = Runs.GetShared(token);
        Assert.NotNull(byToken);
        Assert.Equal(id, byToken!.Id);

        Assert.Null(Runs.GetShared("not-a-real-token"));
    }

    [Fact]
    public void An_unshared_run_has_no_token_to_find_it_by()
    {
        var id = Runs.Save("owner-a", SampleMlp());

        // Never shared, so no token exists at all — GetShared has nothing to match, for any input.
        Assert.Null(Runs.GetShared(id)); // the id itself is not a valid token
    }
}
