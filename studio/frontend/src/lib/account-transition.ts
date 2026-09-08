// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const BROWSER_ACCOUNT_KEY = "unsloth.browser-account.v1";

/** Browser chrome only. Never add credentials, content, model choices or profile data. */
export const ACCOUNT_CHROME_KEYS = new Set([
  "theme",
  "palette",
  "unsloth_appearance_customization",
  "unsloth_locale",
  "sidebar_pinned",
  "sidebar_width",
  "chat_settings_width",
  "unsloth_sidebar_navigate_open",
  "unsloth_settings_active_tab",
  "unsloth_loaded_models_collapsed",
  "unsloth_loaded_models_dismissed",
  "unsloth-rag-preview-width",
]);
// Version-specific notice dismissals contain no account data.
export const ACCOUNT_CHROME_PREFIXES = [
  "unsloth_web_update_dismissed:",
] as const;
export const ACCOUNT_DATABASES = [
  "unsloth-data-recipes",
  "unsloth-data-recipe-executions",
] as const;

export type AccountTransitionBrowser = Pick<
  Window,
  "localStorage" | "indexedDB" | "location"
>;

/**
 * Who the browser now belongs to. Usernames can be renamed and recreated, so the immutable
 * `accountId` decides whether this browser's data carries over; it is absent only against a
 * server too old to send it.
 */
export type BrowserAccount = { username: string; accountId?: string | null };

const ACCOUNT_ID_MARKER_PREFIX = "account:";

/**
 * The name the backend stores. `auth/storage.py` casefolds, then requires
 * `[a-z0-9_-]{3,32}`, so every stored name is ASCII and lowercase says the same thing.
 * A name that survives this and still does not exist server-side simply fails to log in.
 */
export function normalizeAccountUsername(username: string): string {
  return username.trim().toLowerCase();
}

/**
 * The value stored under {@link BROWSER_ACCOUNT_KEY}: `account:<id>:<username>` once the server
 * supplies an id, else the bare normalized username, so markers from either build stay comparable.
 */
export function browserAccountMarker(account: BrowserAccount | string): string {
  const identity: BrowserAccount =
    typeof account === "string" ? { username: account } : account;
  const username = normalizeAccountUsername(identity.username);
  if (!username) throw new Error("Missing account username.");
  return identity.accountId
    ? `${ACCOUNT_ID_MARKER_PREFIX}${identity.accountId}:${username}`
    : username;
}

type MarkedAccount = { accountId: string | null; username: string };
function parseAccountMarker(marker: string): MarkedAccount {
  const legacy = {
    accountId: null,
    username: normalizeAccountUsername(marker),
  };
  if (!marker.startsWith(ACCOUNT_ID_MARKER_PREFIX)) return legacy;
  const qualified = marker.slice(ACCOUNT_ID_MARKER_PREFIX.length);
  const separator = qualified.indexOf(":");
  // A marker no build wrote stays whole, so it can only compare as different.
  if (separator <= 0 || separator === qualified.length - 1) return legacy;
  return {
    accountId: qualified.slice(0, separator),
    username: normalizeAccountUsername(qualified.slice(separator + 1)),
  };
}

/**
 * Whether the browser's data may carry over. Ids decide it when both sides have one; falling back
 * to the reusable username cannot tell a recreated account apart.
 */
function isSameAccount(previous: MarkedAccount, next: MarkedAccount): boolean {
  if (previous.accountId && next.accountId)
    return previous.accountId === next.accountId;
  return previous.username === next.username;
}

export function resetFullAccessForMultiUser(storage: Storage): void {
  if (storage.getItem("unsloth_chat_permission_mode") === "full") {
    storage.setItem("unsloth_chat_permission_mode", "auto");
  }
}

function deleteAccountDatabase(
  indexedDB: IDBFactory,
  name: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onsuccess = () => resolve();
    request.onerror = () =>
      reject(request.error ?? new Error("Could not clear account data."));
    request.onblocked = () =>
      reject(
        new Error(
          "Close other Unsloth tabs and retry signing in to clear the previous account's data.",
        ),
      );
  });
}

/**
 * Run before publishing new tokens; an absent marker means the historical owner browser. The marker
 * is published last so other tabs reload only once the new session is ready. Returns true when a
 * document navigation replaces every hydrated store.
 */
export async function transitionBrowserAccount(
  account: BrowserAccount | string,
  postAuthRoute: string,
  commitSession: () => void,
  browser: AccountTransitionBrowser = window,
): Promise<boolean> {
  const marker = browserAccountMarker(account);
  const storage = browser.localStorage;
  const changed = !isSameAccount(
    parseAccountMarker(storage.getItem(BROWSER_ACCOUNT_KEY) ?? "unsloth"),
    parseAccountMarker(marker),
  );
  if (changed) {
    const keys = Array.from({ length: storage.length }, (_, index) =>
      storage.key(index),
    );
    for (const key of keys) {
      if (
        !key ||
        key === BROWSER_ACCOUNT_KEY ||
        ACCOUNT_CHROME_KEYS.has(key) ||
        ACCOUNT_CHROME_PREFIXES.some((prefix) => key.startsWith(prefix))
      )
        continue;
      if (key.startsWith("unsloth") || key.startsWith("chat-draft"))
        storage.removeItem(key);
    }
    await Promise.all(
      ACCOUNT_DATABASES.map((name) =>
        deleteAccountDatabase(browser.indexedDB, name),
      ),
    );
  }
  commitSession();
  if (storage.getItem(BROWSER_ACCOUNT_KEY) !== marker)
    storage.setItem(BROWSER_ACCOUNT_KEY, marker);
  if (changed) browser.location.replace(postAuthRoute);
  return changed;
}

/** One listener and at most one reload per document, including duplicate storage events. */
const watchedBrowsers = new WeakSet<Window>();
export function installAccountTransitionListener(
  browser: Window = window,
): void {
  if (watchedBrowsers.has(browser)) return;
  watchedBrowsers.add(browser);
  let reloading = false;
  browser.addEventListener("storage", (event) => {
    if (
      reloading ||
      event.key !== BROWSER_ACCOUNT_KEY ||
      event.newValue === null
    )
      return;
    if (event.storageArea && event.storageArea !== browser.localStorage) return;
    const previous = parseAccountMarker(event.oldValue ?? "unsloth");
    if (isSameAccount(previous, parseAccountMarker(event.newValue))) return;
    reloading = true;
    browser.location.reload();
  });
}
