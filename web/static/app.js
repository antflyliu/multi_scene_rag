const statusText = document.getElementById("statusText");
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const chatSendBtn = document.getElementById("chatSendBtn");
const maskedApiKey = document.getElementById("maskedApiKey");
const apiKeyInput = document.getElementById("apiKeyInput");
const llmVendorInput = document.getElementById("llmVendorInput");
const baseUrlInput = document.getElementById("baseUrlInput");
const modelNameInput = document.getElementById("modelNameInput");
const embeddingProviderInput = document.getElementById("embeddingProviderInput");
const embeddingProviderCustomRow = document.getElementById("embeddingProviderCustomRow");
const embeddingProviderCustomInput = document.getElementById("embeddingProviderCustomInput");
const embeddingProviderBackBtn = document.getElementById("embeddingProviderBackBtn");
const embeddingModelInput = document.getElementById("embeddingModelInput");
const embeddingModelCustomRow = document.getElementById("embeddingModelCustomRow");
const embeddingModelCustomInput = document.getElementById("embeddingModelCustomInput");
const embeddingModelBackBtn = document.getElementById("embeddingModelBackBtn");
const embeddingSourceInput = document.getElementById("embeddingSourceInput");
const embeddingSourceCustomRow = document.getElementById("embeddingSourceCustomRow");
const embeddingSourceCustomInput = document.getElementById("embeddingSourceCustomInput");
const embeddingSourceBackBtn = document.getElementById("embeddingSourceBackBtn");
const embeddingDeviceInput = document.getElementById("embeddingDeviceInput");
const embeddingDeviceCustomRow = document.getElementById("embeddingDeviceCustomRow");
const embeddingDeviceCustomInput = document.getElementById("embeddingDeviceCustomInput");
const embeddingDeviceBackBtn = document.getElementById("embeddingDeviceBackBtn");
const embeddingModelCheckCard = document.getElementById("embeddingModelCheckCard");
const checkEmbeddingModelBtn = document.getElementById("checkEmbeddingModelBtn");
const embeddingModelStatusText = document.getElementById("embeddingModelStatusText");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const sceneTableBody = document.getElementById("sceneTableBody");
const addSceneBtn = document.getElementById("addSceneBtn");
const saveScenesBtn = document.getElementById("saveScenesBtn");
const rebuildAllBtn = document.getElementById("rebuildAllBtn");
const rebuildTaskCard = document.getElementById("rebuildTaskCard");
const taskStatusBadge = document.getElementById("taskStatusBadge");
const taskIdText = document.getElementById("taskIdText");
const taskSceneText = document.getElementById("taskSceneText");
const taskProgressBar = document.getElementById("taskProgressBar");
const taskMessageText = document.getElementById("taskMessageText");
let rebuildInProgress = false;
let embeddingCapabilities = {
  fields: {},
  models_by_provider: {},
};
const LLM_VENDOR_BASE_URLS = {
  dashscope: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  openai: "https://api.openai.com/v1",
  claude: "",
  gemini: "",
  glm: "",
  kimi: "",
  deepseek: "https://api.deepseek.com/v1",
  custom: "",
};

function setStatus(message) {
  statusText.textContent = message;
}

function setRebuildUIState(isBusy) {
  rebuildInProgress = isBusy;
  chatSendBtn.disabled = isBusy;
  rebuildAllBtn.disabled = isBusy;
  saveScenesBtn.disabled = isBusy;
  document.querySelectorAll(".rebuild-scene-btn").forEach((btn) => {
    btn.disabled = isBusy;
  });
}

function updateTaskCard(task) {
  if (!task) {
    rebuildTaskCard.classList.add("hidden");
    return;
  }

  rebuildTaskCard.classList.remove("hidden");
  taskIdText.textContent = task.task_id || "-";
  taskSceneText.textContent = task.scene_key || "-";
  taskStatusBadge.textContent = task.status || "unknown";
  taskStatusBadge.classList.remove("running", "completed", "failed");
  if (task.status === "running") taskStatusBadge.classList.add("running");
  if (task.status === "completed") taskStatusBadge.classList.add("completed");
  if (task.status === "failed") taskStatusBadge.classList.add("failed");
  const progress = Number(task.progress || 0);
  taskProgressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
  const stageText = (task.stage || "").trim();
  const currentScene = (task.current_scene || "").trim();
  const sceneIndex = Number(task.scene_index || 0);
  const totalScenes = Number(task.total_scenes || 0);
  const elapsedSeconds = Number(task.elapsed_seconds || 0);
  const parts = [];
  if (task.message) parts.push(task.message);
  if (totalScenes > 0) {
    const sceneLabel = currentScene ? `${currentScene}` : "-";
    parts.push(`进度场景 ${Math.min(sceneIndex, totalScenes)}/${totalScenes}（当前: ${sceneLabel}）`);
  }
  if (stageText) parts.push(`阶段: ${stageText}`);
  if (elapsedSeconds > 0) parts.push(`耗时: ${elapsedSeconds.toFixed(2)}s`);
  taskMessageText.textContent = parts.join(" | ");
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return resp;
  } finally {
    clearTimeout(timer);
  }
}

function formatUsageText(usage) {
  if (!usage || typeof usage !== "object") {
    return "";
  }
  const input = Number(usage.input_tokens || 0);
  const output = Number(usage.output_tokens || 0);
  const total = Number(usage.total_tokens || 0);
  return `Token 消耗：输入 ${input} / 输出 ${output} / 总计 ${total}`;
}

function appendMessage(role, text, usage = null, meta = null) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const content = document.createElement("div");
  content.className = "msg-text";
  content.textContent = text;
  div.appendChild(content);
  if (role === "bot") {
    if (meta && meta.answer_mode) {
      const metaRow = document.createElement("div");
      metaRow.className = "msg-meta";
      const modeLabel = document.createElement("span");
      modeLabel.className = `msg-mode-badge msg-mode-${meta.answer_mode}`;
      modeLabel.textContent = meta.answer_mode === "structured" ? "📊 确定性答案" : "🔍 RAG";
      metaRow.appendChild(modeLabel);
      if (meta.answered_scene) {
        const sceneLabel = document.createElement("span");
        sceneLabel.className = "msg-scene-label";
        sceneLabel.textContent = `场景: ${meta.answered_scene}`;
        metaRow.appendChild(sceneLabel);
      }
      div.appendChild(metaRow);
    }
    const usageText = formatUsageText(usage);
    if (usageText) {
      const usageLine = document.createElement("div");
      usageLine.className = "msg-usage";
      usageLine.textContent = usageText;
      div.appendChild(usageLine);
    }
  }
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.remove("active");
  });
  const target = document.getElementById(`tab-${tabName}`);
  if (target) target.classList.add("active");
}

function normalizeVendor(value) {
  return (value || "").trim().toLowerCase();
}

function isPresetUrl(value) {
  const target = value || "";
  return Object.values(LLM_VENDOR_BASE_URLS).some((url) => url && url === target);
}

function normalizeEmbeddingProvider(value) {
  return (value || "").trim().toLowerCase();
}

function normalizeFieldValue(value) {
  return (value || "").trim();
}

function buildEmbeddingModelOptionLabel(item) {
  if (!item) return "";
  if (item.description) return `${item.label} - ${item.description}`;
  return item.label || item.value || "";
}

function setCustomMode(selectElement, customRowElement, customInput, useCustom, customValue = "") {
  selectElement.classList.toggle("hidden", useCustom);
  customRowElement.classList.toggle("hidden", !useCustom);
  if (useCustom) {
    customInput.value = normalizeFieldValue(customValue);
    customInput.focus();
  } else {
    customInput.value = "";
  }
}

function getFieldOptions(fieldName) {
  return embeddingCapabilities.fields?.[fieldName]?.options || [];
}

function getFieldDefault(fieldName, fallback = "") {
  return embeddingCapabilities.fields?.[fieldName]?.default || fallback;
}

function isFieldCustomizable(fieldName) {
  return Boolean(embeddingCapabilities.fields?.[fieldName]?.customizable);
}

function getEmbeddingModelChoices(provider) {
  const normalizedProvider = normalizeEmbeddingProvider(provider);
  const modelMap = embeddingCapabilities.models_by_provider || {};
  if (normalizedProvider && modelMap[normalizedProvider]) {
    return modelMap[normalizedProvider];
  }
  const defaultProvider = normalizeEmbeddingProvider(getFieldDefault("embedding_provider", "dashscope"));
  return modelMap[defaultProvider] || [];
}

function populateSelectWithCustom(
  selectElement,
  customRowElement,
  customInput,
  options,
  selectedValue,
  customizable
) {
  const targetValue = normalizeFieldValue(selectedValue);
  const hasTargetInChoices = options.some((item) => item.value === targetValue);
  const defaultValue = options[0]?.value || "";
  const useCustom = Boolean(customizable && targetValue && !hasTargetInChoices);

  selectElement.innerHTML = "";
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = buildEmbeddingModelOptionLabel(item);
    selectElement.appendChild(option);
  });

  if (customizable) {
    const customOption = document.createElement("option");
    customOption.value = "__custom__";
    customOption.textContent = "自定义";
    selectElement.appendChild(customOption);
  }

  if (useCustom) {
    selectElement.value = "__custom__";
    setCustomMode(selectElement, customRowElement, customInput, true, targetValue);
    return;
  }

  selectElement.value = targetValue || defaultValue || (customizable ? "__custom__" : "");
  if (selectElement.value === "__custom__") {
    setCustomMode(selectElement, customRowElement, customInput, true, targetValue);
  } else {
    setCustomMode(selectElement, customRowElement, customInput, false);
  }
}

function resolveSelectValue(selectElement, customRowElement, customInput) {
  if (!customRowElement.classList.contains("hidden")) {
    return normalizeFieldValue(customInput.value);
  }
  return normalizeFieldValue(selectElement.value);
}

function pickExistingOrFirst(selectElement, preferredValue = "") {
  const options = Array.from(selectElement.options || []);
  const hasPreferred = options.some((opt) => opt.value === preferredValue);
  if (hasPreferred) {
    return preferredValue;
  }
  return options.find((opt) => opt.value !== "__custom__")?.value || "__custom__";
}

function getCurrentEmbeddingConfig() {
  return {
    embedding_provider: resolveSelectValue(
      embeddingProviderInput,
      embeddingProviderCustomRow,
      embeddingProviderCustomInput
    ),
    embedding_model: resolveSelectValue(
      embeddingModelInput,
      embeddingModelCustomRow,
      embeddingModelCustomInput
    ),
    embedding_source: resolveSelectValue(
      embeddingSourceInput,
      embeddingSourceCustomRow,
      embeddingSourceCustomInput
    ),
    embedding_device: resolveSelectValue(
      embeddingDeviceInput,
      embeddingDeviceCustomRow,
      embeddingDeviceCustomInput
    ),
  };
}

function updateEmbeddingCheckVisibility() {
  const provider = normalizeEmbeddingProvider(getCurrentEmbeddingConfig().embedding_provider);
  const isLocal = provider === "local";
  embeddingModelCheckCard.classList.toggle("hidden", !isLocal);
  if (!isLocal) {
    embeddingModelStatusText.textContent = "当前 provider 非 local，无需本地模型检查。";
  }
}

function setEmbeddingStatusText(message) {
  embeddingModelStatusText.textContent = message || "未检查";
}

async function requestEmbeddingModelStatus() {
  const payload = getCurrentEmbeddingConfig();
  const resp = await fetchWithTimeout("/api/embedding/model-status", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "检查模型状态失败");
  }
  return data;
}

async function requestEmbeddingModelDownload() {
  const payload = getCurrentEmbeddingConfig();
  const resp = await fetchWithTimeout("/api/embedding/model-download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "启动下载模型任务失败");
  }
  return data;
}

async function requestEmbeddingDownloadStatus(taskId) {
  const resp = await fetchWithTimeout(`/api/embedding/model-download-status/${taskId}`, {}, 30000);
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "获取下载任务状态失败");
  }
  return data;
}

async function pollEmbeddingDownloadTask(taskId) {
  while (true) {
    const data = await requestEmbeddingDownloadStatus(taskId);
    const progress = Number(data.progress || 0);
    const message = data.message || "模型下载中...";
    setEmbeddingStatusText(`${message}（${progress}%）`);
    if (data.status === "completed") {
      return data;
    }
    if (data.status === "failed") {
      throw new Error(data.error || data.message || "模型下载失败");
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

async function confirmEmbeddingModelStatusAfterSave() {
  const provider = normalizeEmbeddingProvider(getCurrentEmbeddingConfig().embedding_provider);
  if (provider !== "local") {
    return;
  }

  const statusResult = await requestEmbeddingModelStatus();
  setEmbeddingStatusText(statusResult.message || "模型状态检查完成。");

  if (!statusResult.available && statusResult.needs_download) {
    const targetPath = statusResult.resolved_path || "配置目录";
    const shouldDownload = window.confirm(
      `模型未下载。建议先下载模型到：${targetPath}。是否立即下载？`
    );
    if (!shouldDownload) {
      setEmbeddingStatusText("模型未下载，请先下载模型。");
      return;
    }
    setEmbeddingStatusText("正在下载模型，请稍候...");
    checkEmbeddingModelBtn.disabled = true;
    try {
      const startResult = await requestEmbeddingModelDownload();
      const taskId = startResult.task_id;
      if (!taskId) {
        throw new Error("下载任务创建失败，缺少 task_id");
      }
      const finalResult = await pollEmbeddingDownloadTask(taskId);
      setEmbeddingStatusText(finalResult.message || "模型下载完成。");
    } finally {
      checkEmbeddingModelBtn.disabled = false;
    }
  }
}

function renderEmbeddingProviderOptions(selectedProvider = "") {
  populateSelectWithCustom(
    embeddingProviderInput,
    embeddingProviderCustomRow,
    embeddingProviderCustomInput,
    getFieldOptions("embedding_provider"),
    selectedProvider || getFieldDefault("embedding_provider", "dashscope"),
    isFieldCustomizable("embedding_provider")
  );
}

function renderEmbeddingModelOptions(selectedModel = "") {
  const provider = resolveSelectValue(
    embeddingProviderInput,
    embeddingProviderCustomRow,
    embeddingProviderCustomInput
  );
  populateSelectWithCustom(
    embeddingModelInput,
    embeddingModelCustomRow,
    embeddingModelCustomInput,
    getEmbeddingModelChoices(provider),
    selectedModel,
    true
  );
}

function renderEmbeddingSourceOptions(selectedSource = "") {
  populateSelectWithCustom(
    embeddingSourceInput,
    embeddingSourceCustomRow,
    embeddingSourceCustomInput,
    getFieldOptions("embedding_source"),
    selectedSource || getFieldDefault("embedding_source", "huggingface"),
    isFieldCustomizable("embedding_source")
  );
}

function renderEmbeddingDeviceOptions(selectedDevice = "") {
  populateSelectWithCustom(
    embeddingDeviceInput,
    embeddingDeviceCustomRow,
    embeddingDeviceCustomInput,
    getFieldOptions("embedding_device"),
    selectedDevice || getFieldDefault("embedding_device", "cpu"),
    isFieldCustomizable("embedding_device")
  );
}

function renderAllEmbeddingSelectors(settings = {}) {
  renderEmbeddingProviderOptions(settings.embedding_provider || "");
  renderEmbeddingModelOptions(settings.embedding_model || "");
  renderEmbeddingSourceOptions(settings.embedding_source || "");
  renderEmbeddingDeviceOptions(settings.embedding_device || "");
  updateEmbeddingCheckVisibility();
  setEmbeddingStatusText("未检查");
}

function onGenericSelectChanged(selectElement, customRowElement, customInput) {
  if (selectElement.value === "__custom__") {
    setCustomMode(selectElement, customRowElement, customInput, true, "");
  } else {
    setCustomMode(selectElement, customRowElement, customInput, false);
  }
}

function onEmbeddingConfigChanged() {
  updateEmbeddingCheckVisibility();
  setEmbeddingStatusText("未检查");
}

async function loadEmbeddingCapabilities() {
  const resp = await fetchWithTimeout("/api/embedding-capabilities");
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "获取 Embedding 能力配置失败");
  }
  embeddingCapabilities = {
    fields: data.fields || {},
    models_by_provider: data.models_by_provider || {},
  };
}

function applyVendorPreset({ forceBaseUrl = false } = {}) {
  const vendor = normalizeVendor(llmVendorInput.value) || "custom";
  const presetUrl = LLM_VENDOR_BASE_URLS[vendor] || "";
  const isCustom = vendor === "custom";
  baseUrlInput.disabled = !isCustom && Boolean(presetUrl);
  if (presetUrl && (forceBaseUrl || !baseUrlInput.value.trim())) {
    baseUrlInput.value = presetUrl;
  }
  if (!presetUrl && !isCustom && forceBaseUrl) {
    const currentBaseUrl = baseUrlInput.value.trim();
    if (isPresetUrl(currentBaseUrl)) {
      baseUrlInput.value = "";
    }
  }
  if (!presetUrl && !isCustom && !baseUrlInput.value.trim()) {
    baseUrlInput.placeholder = "请填写该厂商的 OpenAI 兼容 Base URL";
  } else {
    baseUrlInput.placeholder = "";
  }
}

function createSceneRow(sceneKey = "", scene = { name: "", keywords: [], path: "" }) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input class="scene-key" value="${sceneKey}" placeholder="例如: hr" /></td>
    <td><input class="scene-name" value="${scene.name || ""}" placeholder="场景名称" /></td>
    <td><input class="scene-keywords" value="${(scene.keywords || []).join(",")}" placeholder="关键词1,关键词2" /></td>
    <td><input class="scene-path" value="${scene.path || ""}" placeholder="./data/xxx" /></td>
    <td>
      <button class="rebuild-scene-btn">重建</button>
      <button class="danger-btn remove-scene-btn">删除</button>
    </td>
  `;
  tr.querySelector(".remove-scene-btn").addEventListener("click", () => tr.remove());
  tr.querySelector(".rebuild-scene-btn").addEventListener("click", async () => {
    const currentSceneKey = tr.querySelector(".scene-key").value.trim().toLowerCase();
    if (!currentSceneKey) {
      setStatus("请先填写 scene_key，再执行重建");
      return;
    }
    try {
      setStatus(`正在增量重建: ${currentSceneKey} ...`);
      await rebuildScene(currentSceneKey);
    } catch (err) {
      setStatus(`增量重建失败: ${err.message}`);
    }
  });
  sceneTableBody.appendChild(tr);
}

function collectScenesFromTable() {
  const scenes = {};
  sceneTableBody.querySelectorAll("tr").forEach((tr) => {
    const key = tr.querySelector(".scene-key").value.trim().toLowerCase();
    const name = tr.querySelector(".scene-name").value.trim();
    const keywordsRaw = tr.querySelector(".scene-keywords").value.trim();
    const path = tr.querySelector(".scene-path").value.trim();
    if (!key || !name || !path) {
      return;
    }
    const keywords = keywordsRaw
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    scenes[key] = { name, keywords, path };
  });
  return scenes;
}

async function loadSettings() {
  await loadEmbeddingCapabilities();
  const resp = await fetchWithTimeout("/api/settings");
  const data = await resp.json();
  maskedApiKey.value = data.api_key || "";
  llmVendorInput.value = data.llm_vendor || "dashscope";
  baseUrlInput.value = data.base_url || "";
  modelNameInput.value = data.model_name || "";
  renderAllEmbeddingSelectors({
    embedding_provider: data.embedding_provider || "dashscope",
    embedding_model: data.embedding_model || "text-embedding-v1",
    embedding_source: data.embedding_source || "huggingface",
    embedding_device: data.embedding_device || "cpu",
  });
  applyVendorPreset();
}

async function saveSettings() {
  const vendor = normalizeVendor(llmVendorInput.value);
  const presetUrl = LLM_VENDOR_BASE_URLS[vendor] || "";
  if (!presetUrl && vendor !== "custom") {
    const currentBaseUrl = baseUrlInput.value.trim();
    if (!currentBaseUrl || isPresetUrl(currentBaseUrl)) {
      throw new Error("请选择厂商后填写对应 Base URL");
    }
  }
  const payload = {
    llm_vendor: vendor || "custom",
    base_url: baseUrlInput.value.trim(),
    model_name: modelNameInput.value.trim(),
    ...getCurrentEmbeddingConfig(),
  };
  const newApiKey = apiKeyInput.value.trim();
  payload.api_key = newApiKey ? newApiKey : null;

  const resp = await fetchWithTimeout("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "保存设置失败");
  }
  maskedApiKey.value = data.api_key || "";
  apiKeyInput.value = "";
  return data;
}

async function loadScenes() {
  const resp = await fetchWithTimeout("/api/scenes");
  const scenes = await resp.json();
  sceneTableBody.innerHTML = "";
  Object.entries(scenes).forEach(([sceneKey, scene]) => {
    createSceneRow(sceneKey, scene);
  });
}

async function fetchCurrentScenes() {
  const resp = await fetchWithTimeout("/api/scenes");
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "获取当前场景失败");
  }
  return data;
}

async function saveScenes() {
  const scenes = collectScenesFromTable();
  const resp = await fetchWithTimeout("/api/scenes", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenes }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "保存场景失败");
  }
}

async function startRebuildTask(sceneKey) {
  const resp = await fetchWithTimeout("/api/rebuild-index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_key: sceneKey }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "重建索引失败");
  }
  updateTaskCard({
    task_id: data.task_id,
    scene_key: sceneKey,
    status: data.status || "queued",
    progress: 0,
    message: data.message || "任务已启动",
  });
  return data.task_id;
}

async function pollRebuildTask(taskId) {
  while (true) {
    const resp = await fetchWithTimeout(`/api/rebuild-status/${taskId}`, {}, 30000);
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "获取重建状态失败");
    }
    updateTaskCard(data);
    setStatus(`${data.message}（${data.progress}%）`);
    if (data.status === "completed") {
      return data;
    }
    if (data.status === "failed") {
      throw new Error(data.error || "重建任务失败");
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

async function rebuildAll() {
  if (rebuildInProgress) {
    setStatus("已有重建任务进行中，请稍后");
    return;
  }
  setRebuildUIState(true);
  try {
    const taskId = await startRebuildTask("all");
    const result = await pollRebuildTask(taskId);
    setStatus(result.message || "全量重建完成");
    updateTaskCard(result);
  } finally {
    setRebuildUIState(false);
  }
}

async function rebuildScene(sceneKey) {
  if (rebuildInProgress) {
    setStatus("已有重建任务进行中，请稍后");
    return;
  }
  setRebuildUIState(true);
  try {
    const taskId = await startRebuildTask(sceneKey);
    const result = await pollRebuildTask(taskId);
    setStatus(result.message || `增量重建完成: ${sceneKey}`);
    updateTaskCard(result);
  } finally {
    setRebuildUIState(false);
  }
}

async function sendChat() {
  const query = chatInput.value.trim();
  if (!query) return;
  appendMessage("user", query);
  chatInput.value = "";
  setStatus("正在请求回答...");

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      if (resp.status === 409) {
        throw new Error(data.detail || "索引重建中，请稍后再试");
      }
      throw new Error(data.detail || "问答失败");
    }
    const meta = {
      answer_mode: data.answer_mode || "rag",
      answered_scene: data.answered_scene || "",
    };
    appendMessage("bot", data.answer || "", data.usage || null, meta);
    const usageText = formatUsageText(data.usage || null);
    setStatus(usageText || "回答完成");
  } catch (err) {
    appendMessage("bot", `请求失败: ${err.message}`);
    setStatus("请求失败");
  }
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

chatSendBtn.addEventListener("click", () => {
  sendChat().catch((err) => setStatus(`发送失败: ${err.message}`));
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    sendChat().catch((err) => setStatus(`发送失败: ${err.message}`));
  }
});

llmVendorInput.addEventListener("change", () => {
  applyVendorPreset({ forceBaseUrl: true });
});

embeddingProviderInput.addEventListener("change", () => {
  onGenericSelectChanged(embeddingProviderInput, embeddingProviderCustomRow, embeddingProviderCustomInput);
  renderEmbeddingModelOptions(
    resolveSelectValue(embeddingModelInput, embeddingModelCustomRow, embeddingModelCustomInput)
  );
  onEmbeddingConfigChanged();
});

embeddingProviderCustomInput.addEventListener("input", () => {
  if (!embeddingProviderCustomRow.classList.contains("hidden")) {
    renderEmbeddingModelOptions(
      resolveSelectValue(embeddingModelInput, embeddingModelCustomRow, embeddingModelCustomInput)
    );
    onEmbeddingConfigChanged();
  }
});

embeddingModelInput.addEventListener("change", () => {
  onGenericSelectChanged(embeddingModelInput, embeddingModelCustomRow, embeddingModelCustomInput);
  onEmbeddingConfigChanged();
});

embeddingSourceInput.addEventListener("change", () => {
  onGenericSelectChanged(embeddingSourceInput, embeddingSourceCustomRow, embeddingSourceCustomInput);
  onEmbeddingConfigChanged();
});

embeddingDeviceInput.addEventListener("change", () => {
  onGenericSelectChanged(embeddingDeviceInput, embeddingDeviceCustomRow, embeddingDeviceCustomInput);
  onEmbeddingConfigChanged();
});

embeddingProviderBackBtn.addEventListener("click", () => {
  const defaultProvider = pickExistingOrFirst(
    embeddingProviderInput,
    getFieldDefault("embedding_provider", "dashscope")
  );
  embeddingProviderInput.value = defaultProvider;
  setCustomMode(embeddingProviderInput, embeddingProviderCustomRow, embeddingProviderCustomInput, false);
  renderEmbeddingModelOptions(
    resolveSelectValue(embeddingModelInput, embeddingModelCustomRow, embeddingModelCustomInput)
  );
  onEmbeddingConfigChanged();
});

embeddingModelBackBtn.addEventListener("click", () => {
  const defaultModel = getEmbeddingModelChoices(
    resolveSelectValue(embeddingProviderInput, embeddingProviderCustomRow, embeddingProviderCustomInput)
  )[0]?.value || "";
  embeddingModelInput.value = defaultModel || "__custom__";
  if (embeddingModelInput.value === "__custom__") {
    setCustomMode(embeddingModelInput, embeddingModelCustomRow, embeddingModelCustomInput, true, "");
  } else {
    setCustomMode(embeddingModelInput, embeddingModelCustomRow, embeddingModelCustomInput, false);
  }
  onEmbeddingConfigChanged();
});

embeddingSourceBackBtn.addEventListener("click", () => {
  const defaultSource = pickExistingOrFirst(
    embeddingSourceInput,
    getFieldDefault("embedding_source", "huggingface")
  );
  embeddingSourceInput.value = defaultSource;
  setCustomMode(embeddingSourceInput, embeddingSourceCustomRow, embeddingSourceCustomInput, false);
  onEmbeddingConfigChanged();
});

embeddingDeviceBackBtn.addEventListener("click", () => {
  const defaultDevice = pickExistingOrFirst(
    embeddingDeviceInput,
    getFieldDefault("embedding_device", "cpu")
  );
  embeddingDeviceInput.value = defaultDevice;
  setCustomMode(embeddingDeviceInput, embeddingDeviceCustomRow, embeddingDeviceCustomInput, false);
  onEmbeddingConfigChanged();
});

embeddingModelCustomInput.addEventListener("input", onEmbeddingConfigChanged);
embeddingSourceCustomInput.addEventListener("input", onEmbeddingConfigChanged);
embeddingDeviceCustomInput.addEventListener("input", onEmbeddingConfigChanged);

checkEmbeddingModelBtn.addEventListener("click", async () => {
  try {
    checkEmbeddingModelBtn.disabled = true;
    await confirmEmbeddingModelStatusAfterSave();
  } catch (err) {
    setEmbeddingStatusText(`检查失败：${err.message}`);
    setStatus(`模型检查失败: ${err.message}`);
  } finally {
    checkEmbeddingModelBtn.disabled = false;
  }
});

saveSettingsBtn.addEventListener("click", async () => {
  try {
    saveSettingsBtn.disabled = true;
    checkEmbeddingModelBtn.disabled = true;
    const result = await saveSettings();
    if (result.embedding_changed) {
      setStatus(result.reminder || "Embedding 配置已更新，请按需手动重建索引。");
    } else {
      setStatus("设置已保存");
    }
    await confirmEmbeddingModelStatusAfterSave();
  } catch (err) {
    setStatus(`保存设置失败: ${err.message}`);
  } finally {
    saveSettingsBtn.disabled = false;
    checkEmbeddingModelBtn.disabled = false;
  }
});

addSceneBtn.addEventListener("click", () => createSceneRow());

saveScenesBtn.addEventListener("click", async () => {
  try {
    const existingScenes = await fetchCurrentScenes();
    const existingKeys = new Set(Object.keys(existingScenes));

    const currentDraftScenes = collectScenesFromTable();
    const newSceneKeys = Object.keys(currentDraftScenes).filter((key) => !existingKeys.has(key));

    await saveScenes();
    setStatus("场景已保存");

    if (newSceneKeys.length > 0) {
      const shouldRebuild = window.confirm(
        `检测到新场景：${newSceneKeys.join(", ")}。是否立即执行增量重建索引？`
      );
      if (shouldRebuild) {
        for (const sceneKey of newSceneKeys) {
          setStatus(`正在增量重建新场景: ${sceneKey} ...`);
          await rebuildScene(sceneKey);
        }
      }
    }

    await loadScenes();
  } catch (err) {
    setStatus(`保存场景失败: ${err.message}`);
  }
});

rebuildAllBtn.addEventListener("click", async () => {
  try {
    setStatus("正在重建索引...");
    await rebuildAll();
    await loadScenes();
  } catch (err) {
    setStatus(`重建失败: ${err.message}`);
  }
});

async function bootstrap() {
  try {
    await loadSettings();
    await loadScenes();
    setStatus("系统就绪");
  } catch (err) {
    setStatus(`初始化失败: ${err.message}`);
  }
}

bootstrap();
