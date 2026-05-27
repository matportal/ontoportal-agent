# Assistant Account Auth

MatPortal assistant users can bring their own Codex or Gemini Antigravity account auth for OpenCode edit runs. These browser-login flows are intended to work with the user's own account, including free accounts where the provider supports them.

## Codex

1. Open the assistant and choose **AI Settings**.
2. Open the **OpenCode** tab.
3. Set **Auth source** to **Imported account auth**.
4. Click **Connect Codex**.
5. A Codex device-login page opens. Enter the shown one-time code.
6. Leave the settings dialog open until MatPortal reports that Codex auth was saved.

The backend starts `codex login --device-auth` in a private temporary `CODEX_HOME`, waits for `auth.json`, stores it encrypted in the user's assistant settings, and deletes the temporary directory.

## Gemini Antigravity

Gemini Antigravity uses a Google account through OpenCode and does not require the user to bring a paid Vertex account.

1. Open the assistant and choose **AI Settings**.
2. Open the **OpenCode** tab.
3. Set **Auth source** to **Imported account auth**.
4. Click **Connect Gemini Antigravity**.
5. Complete Google login in the new browser tab.
6. Google redirects to `http://localhost:51121/oauth-callback`. The page may not load; that is expected outside a local Antigravity session.
7. Copy the final browser URL from the address bar and paste it into **Gemini callback URL or code**.
8. Click **Finish Gemini login**.

The backend exchanges the callback code for Antigravity-compatible Google OAuth tokens, writes the OpenCode `auth.json` shape, stores it encrypted in the user's assistant settings, and never returns the token values to the browser.

## Fallback

The manual OpenCode and Codex auth JSON text areas remain available for recovery and local debugging. Saved secrets are shown only as `Configured`.
