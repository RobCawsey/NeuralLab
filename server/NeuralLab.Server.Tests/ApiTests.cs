using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace NeuralLab.Server.Tests;

/// <summary>
/// <see cref="RunsTests"/> checks the SQL; this checks the HTTP contract in front of it — header
/// parsing, status codes, and the one piece of routing logic worth distrusting on sight: the
/// <c>?shared=&lt;token&gt;</c> branch on <c>GET /api/runs/{id}</c> cross-checking the token's own
/// run id against the id in the URL, so a valid token for run A cannot be used to read run B by
/// swapping the path segment.
/// </summary>
public class ApiTests : IClassFixture<WebApplicationFactory<Program>>, IDisposable
{
    private readonly string _dataDir;
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public ApiTests(WebApplicationFactory<Program> factory)
    {
        _dataDir = Path.Combine(Path.GetTempPath(), "neurallab-api-tests-" + Guid.NewGuid().ToString("N"));
        // `WithWebHostBuilder` returns a *new* factory pointed at this test's own throwaway data
        // directory — every client this test needs, including a second "stranger" identity, has
        // to come from this same factory rather than a plain `new HttpClient()`, because the
        // in-memory TestServer behind it has no real socket for a stray client to connect to.
        _factory = factory.WithWebHostBuilder(builder =>
            builder.ConfigureAppConfiguration((_, config) =>
                config.AddInMemoryCollection(new Dictionary<string, string?> { ["DataDir"] = _dataDir })));
        _client = _factory.CreateClient();
    }

    public void Dispose()
    {
        if (Directory.Exists(_dataDir)) Directory.Delete(_dataDir, recursive: true);
    }

    private static object Body(string net = "mlp", string dataset = "moons") => new
    {
        title = "a run",
        net,
        dataset,
        config = "net=mlp&data=moons",
        finalMetrics = new { trainLoss = 0.1 },
        finalLoss = 0.1,
    };

    [Fact]
    public async Task Saving_without_an_owner_header_is_rejected()
    {
        var res = await _client.PostAsJsonAsync("/api/runs", Body());
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
    }

    [Fact]
    public async Task Save_then_list_then_reopen_round_trips_through_real_HTTP()
    {
        var owner = Guid.NewGuid().ToString();
        _client.DefaultRequestHeaders.Add("X-Owner-Id", owner);

        var saved = await _client.PostAsJsonAsync("/api/runs", Body());
        Assert.Equal(HttpStatusCode.Created, saved.StatusCode);
        var id = (await saved.Content.ReadFromJsonAsync<Dictionary<string, string>>())!["id"];

        var list = await _client.GetFromJsonAsync<List<Dictionary<string, object>>>("/api/runs");
        Assert.Single(list!);

        var reopened = await _client.GetAsync($"/api/runs/{id}");
        Assert.Equal(HttpStatusCode.OK, reopened.StatusCode);
    }

    [Fact]
    public async Task A_different_owner_cannot_list_or_reopen_someone_elses_run()
    {
        _client.DefaultRequestHeaders.Add("X-Owner-Id", Guid.NewGuid().ToString());
        var saved = await _client.PostAsJsonAsync("/api/runs", Body());
        var id = (await saved.Content.ReadFromJsonAsync<Dictionary<string, string>>())!["id"];

        using var strangerClient = _factory.CreateClient();
        strangerClient.DefaultRequestHeaders.Add("X-Owner-Id", Guid.NewGuid().ToString());

        var list = await strangerClient.GetFromJsonAsync<List<object>>("/api/runs");
        Assert.Empty(list!);

        var reopened = await strangerClient.GetAsync($"/api/runs/{id}");
        Assert.Equal(HttpStatusCode.NotFound, reopened.StatusCode);
    }

    [Fact]
    public async Task Share_then_reopen_by_shared_token_works_with_no_owner_header_at_all()
    {
        _client.DefaultRequestHeaders.Add("X-Owner-Id", Guid.NewGuid().ToString());
        var saved = await _client.PostAsJsonAsync("/api/runs", Body());
        var id = (await saved.Content.ReadFromJsonAsync<Dictionary<string, string>>())!["id"];

        var shareRes = await _client.PostAsync($"/api/runs/{id}/share", null);
        Assert.Equal(HttpStatusCode.OK, shareRes.StatusCode);
        var token = (await shareRes.Content.ReadFromJsonAsync<Dictionary<string, string>>())!["token"];

        using var anon = _factory.CreateClient(); // no header at all
        var viaId = await anon.GetAsync($"/api/runs/{id}?shared={token}");
        Assert.Equal(HttpStatusCode.OK, viaId.StatusCode);

        var viaTokenRoute = await anon.GetAsync($"/api/runs/shared/{token}");
        Assert.Equal(HttpStatusCode.OK, viaTokenRoute.StatusCode);
    }

    [Fact]
    public async Task A_valid_token_cannot_be_used_to_read_a_different_runs_id()
    {
        _client.DefaultRequestHeaders.Add("X-Owner-Id", Guid.NewGuid().ToString());
        var savedA = await _client.PostAsJsonAsync("/api/runs", Body());
        var idA = (await savedA.Content.ReadFromJsonAsync<Dictionary<string, string>>())!["id"];
        var savedB = await _client.PostAsJsonAsync("/api/runs", Body());
        var idB = (await savedB.Content.ReadFromJsonAsync<Dictionary<string, string>>())!["id"];

        var shareRes = await _client.PostAsync($"/api/runs/{idA}/share", null);
        var token = (await shareRes.Content.ReadFromJsonAsync<Dictionary<string, string>>())!["token"];

        using var anon = _factory.CreateClient();
        var crossRead = await anon.GetAsync($"/api/runs/{idB}?shared={token}");
        Assert.Equal(HttpStatusCode.NotFound, crossRead.StatusCode);
    }

    [Fact]
    public async Task Health_responds_with_no_owner_header_needed()
    {
        using var anon = _factory.CreateClient();
        var res = await anon.GetAsync("/api/health");
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
    }
}
