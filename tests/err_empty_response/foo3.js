console.log("foo3")
var node = document.createElement("LI");
var textnode = document.createTextNode("foo3");
node.appendChild(textnode);
document.getElementById("ROOT").appendChild(node);