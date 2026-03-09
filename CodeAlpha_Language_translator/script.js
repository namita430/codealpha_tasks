async function translateText() {
    let text = document.getElementById("inputText").value;
    let source = document.getElementById("sourceLang").value;
    let target = document.getElementById("targetLang").value;
    let url = "https://api.mymemory.translated.net/get?q=" + text + "&langpair=" + source + "|" + target;
    let response = await fetch(url);
    let data = await response.json();
    document.getElementById("outputText").value = data.responseData.translatedText;
}
function copyText() {
    let copyText = document.getElementById("outputText");
    copyText.select();
    document.execCommand("copy");
    alert("Text copied!");
}
