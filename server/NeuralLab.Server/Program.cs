using NeuralLab.Server;

var builder = WebApplication.CreateBuilder(args);

// `.data/` sits beside this project, not inside it — runtime state, never source. Matches the
// path `.gitignore`'s own "server/.data/" comment already promised before this file existed.
// Overridable via config so the test project can point every test at its own throwaway
// directory instead of the same file `dotnet run` would be writing to.
var dataDir = builder.Configuration["DataDir"]
    ?? Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, "..", ".data"));
Db.Init(dataDir);

// Dev only. The SPA runs on Vite's own origin (5173) while this API runs on its own (5150) until
// `dotnet publish` unifies them behind one origin and there is nothing left to allow across.
if (builder.Environment.IsDevelopment())
{
    builder.Services.AddCors(options => options.AddDefaultPolicy(policy =>
        policy.WithOrigins("http://localhost:5173").AllowAnyHeader().AllowAnyMethod()));
}

var app = builder.Build();
if (builder.Environment.IsDevelopment()) app.UseCors();

/// <summary>
/// Every run belongs to whichever anonymous id the browser generated for itself on first use.
/// There is no login anywhere in this project — the design document never specifies accounts, and
/// this personal project already trusts the browser as the unit of identity everywhere else (the
/// challenge ladder's progress, every setting in the URL). The header is a bare UUID with no
/// signature and grants no real security; it exists only to make "list mine" mean something.
/// </summary>
static string? OwnerId(HttpRequest req) =>
    req.Headers.TryGetValue("X-Owner-Id", out var v) && Guid.TryParse(v, out _) ? v.ToString() : null;

app.MapGet("/api/health", () => Results.Ok(new { ok = true }));

app.MapPost("/api/runs", (SaveRunRequest body, HttpRequest req) =>
{
    var owner = OwnerId(req);
    if (owner is null) return Results.BadRequest("missing or malformed X-Owner-Id header");
    if (body.Net is not ("mlp" or "som")) return Results.BadRequest("net must be 'mlp' or 'som'");
    var id = Runs.Save(owner, body);
    return Results.Created($"/api/runs/{id}", new { id });
});

app.MapGet("/api/runs", (HttpRequest req) =>
{
    var owner = OwnerId(req);
    if (owner is null) return Results.BadRequest("missing or malformed X-Owner-Id header");
    return Results.Ok(Runs.ListMine(owner));
});

// The plain-id path, owner-scoped — this is "reopen", the one only the run's own browser can use
// without a share link. `?shared=<token>` on the *same* route lets the id-based link a reader
// might paste around still work for a shared run, checked against the token rather than the
// owner header.
app.MapGet("/api/runs/{id}", (string id, string? shared, HttpRequest req) =>
{
    if (shared is not null)
    {
        var byToken = Runs.GetShared(shared);
        return byToken is not null && byToken.Id == id ? Results.Ok(byToken) : Results.NotFound();
    }
    var owner = OwnerId(req);
    if (owner is null) return Results.NotFound();
    var detail = Runs.GetOwned(id, owner);
    return detail is not null ? Results.Ok(detail) : Results.NotFound();
});

// The token-only path — what a bare `?shared=<token>` on the *app's* URL actually resolves
// through, since a shared link carries no run id at all (§8: "everything needed to reproduce a
// screen, and nothing else").
app.MapGet("/api/runs/shared/{token}", (string token) =>
{
    var detail = Runs.GetShared(token);
    return detail is not null ? Results.Ok(detail) : Results.NotFound();
});

app.MapPost("/api/runs/{id}/share", (string id, HttpRequest req) =>
{
    var owner = OwnerId(req);
    if (owner is null) return Results.NotFound();
    var token = Runs.Share(id, owner);
    return token is not null ? Results.Ok(new ShareResult(token)) : Results.NotFound();
});

app.UseDefaultFiles();
app.UseStaticFiles();
app.MapFallbackToFile("index.html");

app.Run();

// Exposed so the test project can host this app in-process (WebApplicationFactory) without a
// separate entry point to keep in sync.
public partial class Program { }
